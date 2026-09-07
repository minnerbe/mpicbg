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
		final FloatArray2D[] d = octave.getD();
		final int w = d[0].width;
		final int h = d[0].height;
		final int n = d.length;
		if (w < 3 || h < 3 || n < 3) return;

		/*
		 * Pass 1: find all strict extrema over the 26 neighbours with a branch-free stencil. Pixels are
		 * mapped to int keys with the same order as the floats (see stencilRow), because C2 turns
		 * Math.max on floats into a long instruction sequence to honour NaN and -0 but on ints into a
		 * single one. For every DoG level and row, C is the row of keys, L and R the row shifted by one
		 * pixel, P/Q = max/min( L, R ) and H/G = max/min( P/Q, C ). A pixel is an extremum iff its key is
		 * above the max or below the min of its neighbours' keys, which is exactly the outcome of
		 * comparing it to each neighbour in turn (NaN pixels can pass here but fail localize()). All loops
		 * run over scratch rows with identical indices, the only shape C2 vectorizes. Rows are kept in a
		 * ring of three per level.
		 */
		final int[][][] C = new int[n][3][w], P = new int[n][3][w], Q = new int[n][3][w], H = new int[n][3][w], G = new int[n][3][w];
		final float[] row = new float[w];
		final int[] L = new int[w], R = new int[w], MX = new int[w], MN = new int[w];
		long[] found = new long[256];
		int nFound = 0;

		for (int l = 0; l < n; ++l)
			for (int y = 0; y < 2; ++y)
				stencilRow(d[l].data, y, w, row, L, R, C[l][y], P[l][y], Q[l][y], H[l][y], G[l][y]);
		for (int y = 1; y < h - 1; ++y) {
			final int s0 = (y - 1) % 3, s1 = y % 3, s2 = (y + 1) % 3;
			for (int l = 0; l < n; ++l)
				stencilRow(d[l].data, y + 1, w, row, L, R, C[l][s2], P[l][s2], Q[l][s2], H[l][s2], G[l][s2]);
			for (int i = 1; i < n - 1; ++i) {
				// the two horizontal neighbours, the rows above and below, and all three rows of the adjacent levels
				foldMax(MX, P[i][s1], H[i][s0], H[i][s2], H[i - 1][s0], H[i - 1][s1], H[i - 1][s2], H[i + 1][s0], H[i + 1][s1], H[i + 1][s2]);
				foldMin(MN, Q[i][s1], G[i][s0], G[i][s2], G[i - 1][s0], G[i - 1][s1], G[i - 1][s2], G[i + 1][s0], G[i + 1][s1], G[i + 1][s2]);
				final int[] c = C[i][s1];
				// extrema are rare: test blocks with a vectorized or-reduction and scan only blocks that contain one
				for (int x0 = 1; x0 < w - 1; x0 += 64) {
					final int x1 = Math.min(x0 + 64, w - 1);
					if (anyExtremum(c, MX, MN, x0, x1) == 0) continue;
					for (int x = x0; x < x1; ++x) {
						if (c[x] > MX[x] || c[x] < MN[x]) {
							if (nFound == found.length) found = Arrays.copyOf(found, 2 * nFound);
							found[nFound++] = ((long)i << 40) | ((long)y << 20) | x;
						}
					}
				}
			}
		}

		// Pass 2: localize the extrema in the order of the original scan (level, row and column descending)
		Arrays.sort(found, 0, nFound);
		for (int k = nFound - 1; k >= 0; --k) {
			final long key = found[k];
			localize(d, (int)(key >>> 40), (int)((key >>> 20) & 0xfffff), (int)(key & 0xfffff));
		}
	}

	/**
	 * Row {@code y} of {@code data} as int keys into {@code C}, the max/min of the horizontal neighbours
	 * into {@code P}/{@code Q}, and the max/min including the pixel itself into {@code H}/{@code G}.
	 * The key of a float is its bit pattern with the magnitude bits flipped for negative values, which
	 * orders like the float; adding 0 first maps -0 onto +0 so that the two compare equal.
	 */
	private static void stencilRow(
			final float[] data, final int y, final int w, final float[] row, final int[] L, final int[] R,
			final int[] C, final int[] P, final int[] Q, final int[] H, final int[] G
	) {
		System.arraycopy(data, y * w, row, 0, w);
		for (int x = 0; x < w; ++x) {
			final int b = Float.floatToRawIntBits(row[x] + 0f);
			C[x] = b ^ ((b >> 31) & 0x7fffffff);
		}
		System.arraycopy(C, 0, L, 1, w - 1); // L[ x ] = C[ x - 1 ]
		System.arraycopy(C, 1, R, 0, w - 1); // R[ x ] = C[ x + 1 ]
		for (int x = 0; x < w; ++x) {
			P[x] = Math.max(L[x], R[x]);
			Q[x] = Math.min(L[x], R[x]);
		}
		for (int x = 0; x < w; ++x) {
			H[x] = Math.max(P[x], C[x]);
			G[x] = Math.min(Q[x], C[x]);
		}
	}

	/** {@code MX = max( P, H1, ..., H8 )}, elementwise. */
	private static void foldMax(final int[] MX, final int[] P, final int[] H1, final int[] H2, final int[] H3, final int[] H4, final int[] H5, final int[] H6, final int[] H7, final int[] H8) {
		for (int x = 0; x < MX.length; ++x)
			MX[x] = Math.max(Math.max(Math.max(Math.max(P[x], H1[x]), Math.max(H2[x], H3[x])), Math.max(Math.max(H4[x], H5[x]), Math.max(H6[x], H7[x]))), H8[x]);
	}

	/** {@code MN = min( Q, G1, ..., G8 )}, elementwise. */
	private static void foldMin(final int[] MN, final int[] Q, final int[] G1, final int[] G2, final int[] G3, final int[] G4, final int[] G5, final int[] G6, final int[] G7, final int[] G8) {
		for (int x = 0; x < MN.length; ++x)
			MN[x] = Math.min(Math.min(Math.min(Math.min(Q[x], G1[x]), Math.min(G2[x], G3[x])), Math.min(Math.min(G4[x], G5[x]), Math.min(G6[x], G7[x]))), G8[x]);
	}

	/**
	 * Non-zero iff some {@code c[ x ] > MX[ x ]} or {@code c[ x ] < MN[ x ]} for {@code from <= x < to}. The two
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

		double e000 = d[iac].data[rac + xa];
		double e100 = d[iac].data[rac + xc];
		double e200 = d[iac].data[rac + xb];
		double e010 = d[iac].data[rc + xa];
		double e110 = d[iac].data[rc + xc];
		double e210 = d[iac].data[rc + xb];
		double e020 = d[iac].data[rbc + xa];
		double e120 = d[iac].data[rbc + xc];
		double e220 = d[iac].data[rbc + xb];
		double e001 = d[ic].data[rac + xa];
		double e101 = d[ic].data[rac + xc];
		double e201 = d[ic].data[rac + xb];
		double e011 = d[ic].data[rc + xa];
		double e111 = d[ic].data[rc + xc];
		double e211 = d[ic].data[rc + xb];
		double e021 = d[ic].data[rbc + xa];
		double e121 = d[ic].data[rbc + xc];
		double e221 = d[ic].data[rbc + xb];
		double e002 = d[ibc].data[rac + xa];
		double e102 = d[ibc].data[rac + xc];
		double e202 = d[ibc].data[rac + xb];
		double e012 = d[ibc].data[rc + xa];
		double e112 = d[ibc].data[rc + xc];
		double e212 = d[ibc].data[rc + xb];
		double e022 = d[ibc].data[rbc + xa];
		double e122 = d[ibc].data[rbc + xc];
		double e222 = d[ibc].data[rbc + xb];

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

				    		e000 = d[iac].data[rac + xa];
				    		e100 = d[iac].data[rac + xc];
				    		e200 = d[iac].data[rac + xb];

							e010 = d[iac].data[rc + xa];
							e110 = d[iac].data[rc + xc];
							e210 = d[iac].data[rc + xb];

							e020 = d[iac].data[rbc + xa];
							e120 = d[iac].data[rbc + xc];
							e220 = d[iac].data[rbc + xb];


							e001 = d[ic].data[rac + xa];
							e101 = d[ic].data[rac + xc];
							e201 = d[ic].data[rac + xb];

							e011 = d[ic].data[rc + xa];
							e111 = d[ic].data[rc + xc];
							e211 = d[ic].data[rc + xb];

							e021 = d[ic].data[rbc + xa];
							e121 = d[ic].data[rbc + xc];
							e221 = d[ic].data[rbc + xb];


							e002 = d[ibc].data[rac + xa];
							e102 = d[ibc].data[rac + xc];
							e202 = d[ibc].data[rac + xb];

							e012 = d[ibc].data[rc + xa];
							e112 = d[ibc].data[rc + xc];
							e212 = d[ibc].data[rc + xb];

							e022 = d[ibc].data[rbc + xa];
							e122 = d[ibc].data[rbc + xc];
							e222 = d[ibc].data[rbc + xb];
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
			    } else isLocalizable = false;
			} while (!isLocalized && isLocalizable && t >= 0);
		    // reject detections that could not be localized properly

			if (!isLocalized) {
//						System.err.println( "Localization failed (x: " + xc + ", y: " + yc + ", i: " + ic + ") => (ox: " + ox + ", oy: " + oy + ", oi: " + oi + ")" );
//						if ( ic < 1 || ic > d.length - 2 )
//							System.err.println( "  Detection outside octave." );
				return;
			}

			// reject detections with very low contrast

			if (Math.abs(e111 + 0.5f * (dx * ox + dy * oy + di * oi)) < MIN_CONTRAST) return;

			// reject edge responses

			final double det = dxx * dyy - dxy * dxy;
		    final double trace = dxx + dyy;
		    if (trace * trace / det > MAX_CURVATURE_RATIO) return;

		candidates.addElement(new double[]{ fx, fy, fi });
	}
}
