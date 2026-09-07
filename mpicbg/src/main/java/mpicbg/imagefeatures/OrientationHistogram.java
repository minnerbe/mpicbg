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

/**
 * Histogram of the gradient orientations in a Gaussian window around a candidate, from which
 * {@link FloatArray2DSIFT} and {@link FloatArray2DMOPS} pick the dominant orientations. An
 * instance holds the scratch buffers of one thread of candidates and is not thread-safe.
 */
final class OrientationHistogram {
	final static int BINS = 36;
	final static int BINS1 = BINS - 1;
	final static double BIN_SIZE = 2.0 * Math.PI / BINS;

	final private float[] histogram_bins = new float[BINS];
	/** the window with a one pixel border, its four shifted copies, and per pixel derivatives, weight, magnitude, orientation, bin */
	private float[] pat = new float[0], patL = new float[0], patR = new float[0], patU = new float[0], patD = new float[0];
	private float[] roiDx = new float[0], roiDy = new float[0], roiWin = new float[0], roiMag = new float[0], roiOri = new float[0];
	private int[] roiBin = new int[0];
	/** separable orientation window: ex[ x ] * ey[ y ], sized to the largest window seen so far */
	private float[] maskX = new float[0], maskY = new float[0];

	/**
	 * Build the orientation histogram of the region around a candidate: the gradient magnitudes of
	 * the level closest to the candidate's scale, weighted by a circular Gaussian window with sigma
	 * 1.5 times that of the candidate, accumulated into {@link #BINS} bins of the orientation.
	 *
	 * @param octave the octave of the candidate
	 * @param c candidate {@code 0=>x, 1=>y, 2=>scale index}
	 * @param octave_sigma the candidate's sigma in the octave
	 *
	 * @return the histogram, a buffer of this instance that the next call overwrites
	 */
	float[] compute(
			final FloatArray2DScaleOctave octave,
			final double[] c,
			final double octave_sigma
	) {
		final float[] histogram_bins = this.histogram_bins;
		Arrays.fill(histogram_bins, 0);

		// create a circular gaussian window with sigma 1.5 times that of the feature.
		// The window is separable: exp( -( dx² + dy² ) / s ) = exp( -dx² / s ) * exp( -dy² / s ), so
		// 2 * size exp calls replace size² (mean 563 per candidate). Rounding differs in the last bit
		// from the direct 2d evaluation of Filter.createGaussianKernelOffset.
		final double mask_sigma = octave_sigma * 1.5;
		final int size = Math.max(3, (int)(2 * Math.round(3 * mask_sigma) + 1));
		final int half_size = size / 2;
		final int w2 = size + 2; // the window plus a one pixel border for the derivatives
		final int patchLength = w2 * w2;
		if (pat.length < patchLength) {
			pat = new float[patchLength];
			patL = new float[patchLength];
			patR = new float[patchLength];
			patU = new float[patchLength];
			patD = new float[patchLength];
			roiDx = new float[patchLength];
			roiDy = new float[patchLength];
			roiWin = new float[patchLength];
			roiMag = new float[patchLength];
			roiOri = new float[patchLength];
			roiBin = new int[patchLength];
			maskX = new float[size];
			maskY = new float[size];
		}
		final float[] roiMag = this.roiMag, roiOri = this.roiOri, ex = this.maskX, ey = this.maskY;
		{
			final double two_sq_sigma = 2 * mask_sigma * mask_sigma;
			final double offset_x = c[0] - Math.floor(c[0]), offset_y = c[1] - Math.floor(c[1]);
			for (int i = 0; i < size; ++i) {
				final double dx = i - half_size - offset_x, dy = i - half_size - offset_y;
				ex[i] = (float)Math.exp(-dx * dx / two_sq_sigma);
				ey[i] = (float)Math.exp(-dy * dy / two_sq_sigma);
			}
		}

		/*
		 * Get the gradients in the window around the keypoint, weighted by the window. Window pixel
		 * ( xi, yi ) lives at ( yi + 1 ) * w2 + xi + 1 of the padded scratch arrays; the padding positions
		 * hold junk that is never read. In the interior, the window is copied with its border into pat and
		 * the derivatives, weighted magnitudes and orientations are evaluated in flat passes over shifted
		 * copies (the loops then have identical indices, which C2 vectorizes; the orientation stays scalar).
		 */
		final FloatArray2DScaleOctave.Gradients src = octave.getGradients((int)Math.round(c[2]));
		final int cx = (int)c[0], cy = (int)c[1];
		if (cx - half_size >= 1 && cx - half_size + size <= src.width - 1 && cy - half_size >= 1 && cy - half_size + size <= src.height - 1) {
			final float[] pat = this.pat, patL = this.patL, patR = this.patR, patU = this.patU, patD = this.patD, dxs = this.roiDx, dys = this.roiDy, win = this.roiWin;
			for (int py = 0; py < w2; ++py)
				System.arraycopy(src.data, (cy - half_size - 1 + py) * src.width + cx - half_size - 1, pat, py * w2, w2);
			System.arraycopy(pat, 0, patL, 1, patchLength - 1); // patL[ m ] = pat[ m - 1 ]
			System.arraycopy(pat, 1, patR, 0, patchLength - 1); // patR[ m ] = pat[ m + 1 ]
			System.arraycopy(pat, 0, patU, w2, patchLength - w2); // patU[ m ] = pat[ m - w2 ]
			System.arraycopy(pat, w2, patD, 0, patchLength - w2); // patD[ m ] = pat[ m + w2 ]
			for (int yi = 0; yi < size; ++yi) {
				final float wy = ey[yi];
				final int m0 = (yi + 1) * w2 + 1;
				for (int xi = 0; xi < size; ++xi)
					win[m0 + xi] = wy * ex[xi];
			}
			for (int m = 0; m < patchLength; ++m) {
				dxs[m] = (patR[m] - patL[m]) / 2;
				dys[m] = (patD[m] - patU[m]) / 2;
			}
			for (int m = 0; m < patchLength; ++m)
				roiMag[m] = FloatArray2DScaleOctave.Gradients.mag(dxs[m], dys[m]) * win[m];
			for (int yi = 0; yi < size; ++yi) {
				final int m0 = (yi + 1) * w2 + 1;
				for (int xi = 0; xi < size; ++xi)
					roiOri[m0 + xi] = Filter.fastAtan2(dys[m0 + xi], dxs[m0 + xi]);
			}
		} else {
			// the window reaches over the image border: clamp coordinates pixel by pixel
			for (int yi = 0; yi < size; ++yi) {
				final int ya = Math.max(0, Math.min(src.height - 1, cy + yi - half_size));
				final int ra_x = Math.min(cx, src.width - 1);
				final float wy = ey[yi];
				final int m0 = (yi + 1) * w2 + 1;
				for (int xi = 0; xi < size; ++xi) {
					final int xa = Math.max(0, Math.min(src.width - 2, ra_x + xi - half_size));
					final float der_x = src.derX(xa, ya), der_y = src.derY(xa, ya);
					roiMag[m0 + xi] = FloatArray2DScaleOctave.Gradients.mag(der_x, der_y) * (wy * ex[xi]);
					roiOri[m0 + xi] = Filter.fastAtan2(der_y, der_x);
				}
			}
		}

		// build an orientation histogram of the region: bins for the whole patch (vectorizes), accumulation in scan order
		final int[] bin = this.roiBin;
		for (int m = 0; m < patchLength; ++m)
			bin[m] = Math.max(0, Math.min(BINS1, (int)((roiOri[m] + Math.PI) / BIN_SIZE)));
		for (int yi = 0; yi < size; ++yi) {
			final int m0 = (yi + 1) * w2 + 1;
			for (int xi = 0; xi < size; ++xi)
				histogram_bins[bin[m0 + xi]] += roiMag[m0 + xi];
		}

		return histogram_bins;
	}
}
