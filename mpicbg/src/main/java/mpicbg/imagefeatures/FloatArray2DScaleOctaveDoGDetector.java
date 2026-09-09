/*-
 * #%L
 * MPICBG Core Library.
 * %%
 * Copyright (C) 2008 - 2025 Stephan Saalfeld et. al.
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 2 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-2.0.html>.
 * #L%
 */
package mpicbg.imagefeatures;

import java.util.Arrays;
import java.util.Vector;

import mpicbg.util.Matrix3x3;


/**
 * Difference Of Gaussian detector on top of a scale space octave as described
 * by Lowe (2004).
 *
 * BibTeX:
 * <pre>
 * &#64;article{Lowe04,
 *   author  = {David G. Lowe},
 *   title   = {Distinctive Image Features from Scale-Invariant Keypoints},
 *   journal = {International Journal of Computer Vision},
 *   year    = {2004},
 *   volume  = {60},
 *   number  = {2},
 *   pages   = {91--110},
 * }
 * </pre>
 *
 * @author Stephan Saalfeld &lt;saalfeld@mpi-cbg.de&gt;
 * @version 0.1b
 */
public class FloatArray2DScaleOctaveDoGDetector
{
	/**
	 * minimal contrast of a candidate
	 */
	private static final float MIN_CONTRAST = 0.025f;

	/**
	 * maximal curvature ratio, higher values allow more edge-like responses
	 */
	private static final float MAX_CURVATURE = 10;
	private static final float MAX_CURVATURE_RATIO = ( MAX_CURVATURE + 1 ) * ( MAX_CURVATURE + 1 ) / MAX_CURVATURE;

	private FloatArray2DScaleOctave octave;

	/**
	 * detected candidates as float triples 0=>x, 1=>y, 2=>scale index
	 */
	private Vector< double[] > candidates;
	public Vector< double[] > getCandidates()
	{
		return candidates;
	}

	/**
	 * Constructor
	 */
	public FloatArray2DScaleOctaveDoGDetector()
	{
		octave = null;
		candidates = null;
	}

	public void run( final FloatArray2DScaleOctave o )
	{
		octave = o;
		candidates = new Vector< double[] >();
		detectCandidates();
	}

	private void detectCandidates()
	{
		// Each octave is a (n, h, w) array of DoG images, where n is the number of levels, h the height and w the width
		// A strict extremum is a pixel that is either above or below all 26 neighbors in the 3x3x3 cube around it
		final FloatArray2D[] d = octave.getD();
		final int w = d[0].width;
		final int h = d[0].height;
		final int n = d.length;
		if (w < 3 || h < 3 || n < 3)
			return;

		/*
		 * Pass 1: find all strict extrema over the 26 neighbors with a branch-free stencil. Pixels are
		 * mapped to int keys with the same order as the floats (see stencilRow), because C2 turns
		 * Math.max on floats into a long instruction sequence to honor NaN and -0 but on ints into a
		 * single one. For every DoG level and row, C is the row of keys, L and R the row shifted by one
		 * pixel, max2/min2 = max/min(L, R) and max3/min3 = max/min(max2/min2, C). A pixel is an extremum iff
		 * its key is above the max or below the min of its neighbors' keys. This is exactly the outcome of
		 * comparing it to each neighbor in turn (NaN pixels can pass here but fail localize()). All loops
		 * run over scratch rows with identical indices, the only shape C2 vectorizes. Rows are kept in a
		 * ring of three per level.
		 */
		final int[][][] keys = new int[n][3][w];
		final int[][][] max2 = new int[n][3][w];
		final int[][][] min2 = new int[n][3][w];
		final int[][][] max3 = new int[n][3][w];
		final int[][][] min3 = new int[n][3][w];

		// Scratch buffers for the current row of floats, the left and right neighbors, and the max/min of the neighbors
		final float[] row = new float[w];
		final int[] L = new int[w];
		final int[] R = new int[w];
		final int[] MX = new int[w];
		final int[] MN = new int[w];
		long[] found = new long[256];
		int nFound = 0;

		// Fill the first two rows of each level as initialization of the ring buffer
		for (int l = 0; l < n; ++l)
			for (int y = 0; y < 2; ++y)
				stencilRow(d[l].data, y, w, row, L, R, keys[l][y], max2[l][y], min2[l][y], max3[l][y], min3[l][y]);

		// Advance one row at a time for all levels
		for (int y = 1; y < h - 1; ++y) {
			final int s0 = (y - 1) % 3; // previous
			final int s1 = y % 3;       // current
			final int s2 = (y + 1) % 3; // next

			// Put the next row into the ring buffer
			for (int l = 0; l < n; ++l)
				stencilRow(d[l].data, y + 1, w, row, L, R, keys[l][s2], max2[l][s2], min2[l][s2], max3[l][s2], min3[l][s2]);

			// Scan the current row for extrema on all levels
			for (int l = 1; l < n - 1; ++l) {
				// Pass the two horizontal neighbors (max/min2), the rows above and below, and all three rows of the adjacent levels (max/min3)
				foldMax(MX, max2[l][s1], max3[l][s0], max3[l][s2], max3[l - 1][s0], max3[l - 1][s1], max3[l - 1][s2], max3[l + 1][s0], max3[l + 1][s1], max3[l + 1][s2]);
				foldMin(MN, min2[l][s1], min3[l][s0], min3[l][s2], min3[l - 1][s0], min3[l - 1][s1], min3[l - 1][s2], min3[l + 1][s0], min3[l + 1][s1], min3[l + 1][s2]);
				final int[] key = keys[l][s1];

				// Extrema are rare: test 3x3x64 blocks with a vectorized or-reduction and scan only blocks that contain one
				for (int xBlockStart = 1; xBlockStart < w - 1; xBlockStart += 64) {
					final int xBlockEnd = Math.min(xBlockStart + 64, w - 1);
					if (anyExtremum(key, MX, MN, xBlockStart, xBlockEnd) == 0)
						continue;

					for (int x = xBlockStart; x < xBlockEnd; ++x) {
						if (key[x] > MX[x] || key[x] < MN[x]) {
							// If an extremum is found, ensure that the found array has enough capacity to hold it
							if (nFound == found.length)
								found = Arrays.copyOf(found, 2 * nFound);

							// Pack the extremum's location into a long with the level in the top 24 bits,
							// the row in the next 20 bits, and the column in the bottom 20 bits
							found[nFound++] = ((long)l << 40) | ((long)y << 20) | x;
						}
					}
				}
			}
		}

		// Pass 2: localize the extrema in the order of the original scan (level, row and column descending)
		Arrays.sort(found, 0, nFound);
		for (int k = nFound - 1; k >= 0; --k) {
			// Unpack the extremum's location from the long and localize it with subpixel accuracy
			final long key = found[k];
			localize(d, (int)(key >>> 40), (int)((key >>> 20) & 0xfffff), (int)(key & 0xfffff));
		}
	}

	/**
	 * Row {@code y} of {@code data} as int keys into {@code keys}, the max/min of the horizontal neighbors
	 * into {@code max2}/{@code min2}, and the max/min including the pixel itself into {@code max3}/{@code min3}.
	 */
	private static void stencilRow(
			final float[] data, final int y, final int w, final float[] row, final int[] L, final int[] R,
			final int[] keys, final int[] max2, final int[] min2, final int[] max3, final int[] min3
	) {
		// Copy the row of floats into a scratch array, then convert them to int keys. The key
		// of a float is its bit pattern with the magnitude bits flipped for negative values,
		// which orders like the float; adding 0 first maps -0 onto +0 so that the two compare equal.
		System.arraycopy(data, y * w, row, 0, w);
		for (int x = 0; x < w; ++x) {
			final int b = Float.floatToRawIntBits(row[x] + 0f);
			keys[x] = b ^ ((b >> 31) & 0x7fffffff);
		}

		// Copy the keys into shifted arrays (left and right neighbors)
		System.arraycopy(keys, 0, L, 1, w - 1); // L[x] = keys[x - 1]
		System.arraycopy(keys, 1, R, 0, w - 1); // R[x] = keys[x + 1]

		// Compute the point-wise max/min of the horizontal neighbors and the pixel itself
		for (int x = 0; x < w; ++x) {
			max2[x] = Math.max(L[x], R[x]);
			min2[x] = Math.min(L[x], R[x]);
		}
		for (int x = 0; x < w; ++x) {
			max3[x] = Math.max(max2[x], keys[x]);
			min3[x] = Math.min(min2[x], keys[x]);
		}
	}

	/** Tree-reduction {@code MX = max( P, H1, ..., H8 )}, elementwise. */
	private static void foldMax(final int[] MX, final int[] P, final int[] H1, final int[] H2, final int[] H3, final int[] H4, final int[] H5, final int[] H6, final int[] H7, final int[] H8) {
		for (int x = 0; x < MX.length; ++x)
			MX[x] = Math.max(Math.max(Math.max(Math.max(P[x], H1[x]), Math.max(H2[x], H3[x])), Math.max(Math.max(H4[x], H5[x]), Math.max(H6[x], H7[x]))), H8[x]);
	}

	/** Tree-reduction {@code MN = min( Q, G1, ..., G8 )}, elementwise. */
	private static void foldMin(final int[] MN, final int[] Q, final int[] G1, final int[] G2, final int[] G3, final int[] G4, final int[] G5, final int[] G6, final int[] G7, final int[] G8) {
		for (int x = 0; x < MN.length; ++x)
			MN[x] = Math.min(Math.min(Math.min(Math.min(Q[x], G1[x]), Math.min(G2[x], G3[x])), Math.min(Math.min(G4[x], G5[x]), Math.min(G6[x], G7[x]))), G8[x]);
	}

	/**
	 * Non-zero iff some {@code c[x] > MX[x]} or {@code c[x] < MN[x]} for {@code from <= x < to}. The two
	 * differences are non-negative and below 2^32, so neither wraps to 0; C2 vectorizes the or-reduction.
	 */
	private static int anyExtremum(final int[] c, final int[] MX, final int[] MN, final int from, final int to) {
		int any = 0;
		for (int x = from; x < to; ++x)
			any |= (Math.max(c[x], MX[x]) - MX[x]) | (MN[x] - Math.min(c[x], MN[x]));
		return any;
	}

	/**
	 * Localize the extremum at ( xc, yc, ic ) with subpixel accuracy and add it to the candidates unless
	 * it cannot be localized, has too low contrast or is an edge response. If it has to be moved for more
	 * than 0.5 in at least one direction, try again there, but maximally 5 times.
	 */
	private void localize(final FloatArray2D[] d, int ic, int yc, int xc) {
		int iac = ic - 1;
		int ibc = ic + 1;
		int rc = yc * d[ic].width;
		int rac = rc - d[ic].width;
		int rbc = rc + d[ic].width;
		int xa = xc - 1;
		int xb = xc + 1;

		//     e000 unused
		double e100 = d[iac].data[rac + xc];
		//     e200 unused
		double e010 = d[iac].data[rc + xa];
		double e110 = d[iac].data[rc + xc];
		double e210 = d[iac].data[rc + xb];
		//     e020 unused
		double e120 = d[iac].data[rbc + xc];
		//     e220 unused
		double e001 = d[ic].data[rac + xa];
		double e101 = d[ic].data[rac + xc];
		double e201 = d[ic].data[rac + xb];
		double e011 = d[ic].data[rc + xa];
		double e111 = d[ic].data[rc + xc];
		double e211 = d[ic].data[rc + xb];
		double e021 = d[ic].data[rbc + xa];
		double e121 = d[ic].data[rbc + xc];
		double e221 = d[ic].data[rbc + xb];
		//     e002 unused
		double e102 = d[ibc].data[rac + xc];
		//     e202 unused
		double e012 = d[ibc].data[rc + xa];
		double e112 = d[ibc].data[rc + xc];
		double e212 = d[ibc].data[rc + xb];
		//     e022 unused
		double e122 = d[ibc].data[rbc + xc];
		//     e222 unused

		boolean isLocalized = false;
		boolean isLocalizable = true;

		double dx;
		double dy;
		double di;

		double dxx;
		double dyy;
		double dii;

		double dxy;
		double dxi;
		double dyi;

		double ox;
		double oy;
		double oi;

		double od = Double.MAX_VALUE;      // offset square distance

		double fx = 0;
		double fy = 0;
		double fi = 0;

		int t = 5; // maximal number of re-localizations
		do {
			--t;

			// derive at (x, y, i) by center of difference
			dx = (e211 - e011) / 2.0f;
			dy = (e121 - e101) / 2.0f;
			di = (e112 - e110) / 2.0f;

			// create hessian at (x, y, i) by laplace
			final double e111_2 = 2.0f * e111;
			dxx = e011 - e111_2 + e211;
			dyy = e101 - e111_2 + e121;
			dii = e110 - e111_2 + e112;

			dxy = (e221 - e021 - e201 + e001) / 4.0f;
			dxi = (e212 - e012 - e210 + e010) / 4.0f;
			dyi = (e122 - e102 - e120 + e100) / 4.0f;

			// invert hessian
			final double det = Matrix3x3.det(dxx, dxy, dxi, dxy, dyy, dyi, dxi, dyi, dii);
			if (det == 0) return;

			final double det1 = 1.0 / det;

			final double hixx = (dyy * dii - dyi * dyi) * det1;
			final double hixy = (dxi * dyi - dxy * dii) * det1;
			final double hixi = (dxy * dyi - dxi * dyy) * det1;
			final double hiyy = (dxx * dii - dxi * dxi) * det1;
			final double hiyi = (dxi * dxy - dxx * dyi) * det1;
			final double hiii = (dxx * dyy - dxy * dxy) * det1;

			// localize
			ox = -hixx * dx - hixy * dy - hixi * di;
			oy = -hixy * dx - hiyy * dy - hiyi * di;
			oi = -hixi * dx - hiyi * dy - hiii * di;


			final double odc = ox * ox + oy * oy + oi * oi;

			if (odc < 2.0f) {
				if ((Math.abs(ox) > 0.5 || Math.abs(oy) > 0.5 || Math.abs(oi) > 0.5) && odc < od) {
					od = odc;

					xc = (int)Math.round(xc + ox);
					yc = (int)Math.round(yc + oy);
					ic = (int)Math.round(ic + oi);

					if (xc < 1 || yc < 1 || ic < 1 || xc > d[0].width - 2 || yc > d[0].height - 2 || ic > d.length - 2)
						isLocalizable = false;
					else {
						xa = xc - 1;
						xb = xc + 1;
						rc = yc * d[ic].width;
						rac = rc - d[ic].width;
						rbc = rc + d[ic].width;
						iac = ic - 1;
						ibc = ic + 1;

						e100 = d[iac].data[rac + xc];
						e010 = d[iac].data[rc + xa];
						e110 = d[iac].data[rc + xc];
						e210 = d[iac].data[rc + xb];
						e120 = d[iac].data[rbc + xc];
						e001 = d[ic].data[rac + xa];
						e101 = d[ic].data[rac + xc];
						e201 = d[ic].data[rac + xb];
						e011 = d[ic].data[rc + xa];
						e111 = d[ic].data[rc + xc];
						e211 = d[ic].data[rc + xb];
						e021 = d[ic].data[rbc + xa];
						e121 = d[ic].data[rbc + xc];
						e221 = d[ic].data[rbc + xb];
						e102 = d[ibc].data[rac + xc];
						e012 = d[ibc].data[rc + xa];
						e112 = d[ibc].data[rc + xc];
						e212 = d[ibc].data[rc + xb];
						e122 = d[ibc].data[rbc + xc];
					}
				} else {
					fx = xc + ox;
					fy = yc + oy;
					fi = ic + oi;

					if (fx < 0 || fy < 0 || fi < 0 || fx > d[0].width - 1 || fy > d[0].height - 1 || fi > d.length - 1)
						isLocalizable = false;
					else
						isLocalized = true;
				}
			} else {
				isLocalizable = false;
			}
		} while (!isLocalized && isLocalizable && t >= 0);
		// reject detections that could not be localized properly

		if (!isLocalized) {
//						System.err.println( "Localization failed (x: " + xc + ", y: " + yc + ", i: " + ic + ") => (ox: " + ox + ", oy: " + oy + ", oi: " + oi + ")" );
//						if ( ic < 1 || ic > d.length - 2 )
//							System.err.println( "  Detection outside octave." );
			return;
		}

		// reject detections with very low contrast

		if (Math.abs(e111 + 0.5f * (dx * ox + dy * oy + di * oi)) < MIN_CONTRAST)
			return;

		// reject edge responses

		final double det = dxx * dyy - dxy * dxy;
		final double trace = dxx + dyy;
		if (trace * trace / det > MAX_CURVATURE_RATIO)
			return;

		candidates.addElement(new double[]{ fx, fy, fi });
	}
}
