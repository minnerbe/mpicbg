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

/**
 * Histogram of the gradient orientations in a Gaussian window around a candidate, from which
 * {@link FloatArray2DSIFT} and {@link FloatArray2DMOPS} pick the dominant orientations.
 */
final class OrientationHistogram {
	final static int BINS = 36;
	final static int BINS1 = BINS - 1;
	final static double BIN_SIZE = 2.0 * Math.PI / BINS;

	/**
	 * Build the orientation histogram of the region around a candidate: the gradient magnitudes of
	 * the level closest to the candidate's scale, weighted by a circular Gaussian window with sigma
	 * 1.5 times that of the candidate, accumulated into {@link #BINS} bins of the orientation.
	 *
	 * @param octave the octave of the candidate
	 * @param c candidate {@code 0=>x, 1=>y, 2=>scale index}
	 * @param octave_sigma the candidate's sigma in the octave
	 *
	 * @return the histogram
	 */
	static float[] compute(
			final FloatArray2DScaleOctave octave,
			final double[] c,
			final double octave_sigma) {
		final float[] histogram_bins = new float[BINS];

		// create a circular gaussian window with sigma 1.5 times that of the feature
		final FloatArray2D gaussianMask =
			Filter.createGaussianKernelOffset(
					octave_sigma * 1.5,
					c[0] - Math.floor(c[0]),
					c[1] - Math.floor(c[1]),
					false);
		//FloatArrayToImagePlus( gaussianMask, "gaussianMask", 0, 0 ).show();

		// get the gradients in a region arround the keypoints location
		final FloatArray2DScaleOctave.Gradients src = octave.getGradients((int)Math.round(c[2]));
		final FloatArray2D[] gradientROI = new FloatArray2D[2];
		gradientROI[0] = new FloatArray2D(gaussianMask.width, gaussianMask.width);
		gradientROI[1] = new FloatArray2D(gaussianMask.width, gaussianMask.width);

		final int half_size = gaussianMask.width / 2;
		int n = gaussianMask.width * gaussianMask.width - 1;
		for (int yi = gaussianMask.width - 1; yi >= 0; --yi) {
			final int ya = Math.max(0, Math.min(src.height - 1, (int)c[1] + yi - half_size));
			final int ra_x = Math.min((int)c[0], src.width - 1);

			for (int xi = gaussianMask.width - 1; xi >= 0; --xi) {
				final int xa = Math.max(0, Math.min(src.width - 2, ra_x + xi - half_size));
				final float der_x = src.derX(xa, ya), der_y = src.derY(xa, ya);
				gradientROI[0].data[n] = FloatArray2DScaleOctave.Gradients.mag(der_x, der_y);
				gradientROI[1].data[n] = (float)Math.atan2(der_y, der_x);
				--n;
			}
		}

		// and mask this region with the precalculated gaussion window
		for (int i = 0; i < gradientROI[0].data.length; ++i) {
			gradientROI[0].data[i] *= gaussianMask.data[i];
		}

		// build an orientation histogram of the region
		for (int i = 0; i < gradientROI[0].data.length; ++i) {
			final int bin = Math.max(0, Math.min(BINS1, (int)((gradientROI[1].data[i] + Math.PI) / BIN_SIZE)));
			histogram_bins[bin] += gradientROI[0].data[i];
		}

		return histogram_bins;
	}
}
