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

import mpicbg.util.Util;

/**
 *
 * @author Stephan Saalfeld &lt;saalfeld@mpi-cbg.de&gt;
 * @version 0.2b
 */
public class Filter
{
	/**
     * Create a 1d-Gaussian kernel of appropriate size.
     *
     * @param sigma Standard deviation of the Gaussian kernel
     * @param normalize Normalize integral of the Gaussian kernel to 1 or not...
     *
     * @return float[] Gaussian kernel of appropriate size
     */
    final static public double[] createGaussianKernel(
    		final double sigma,
    		final boolean normalize )
	{
		double[] kernel;

		if ( sigma <= 0 )
		{
			kernel = new double[ 3 ];
			kernel[ 1 ] = 1;
		}
		else
		{
			final int size = Math.max( 3, ( int ) ( 2 * ( int )( 3 * sigma + 0.5 ) + 1 ) );

			final double two_sq_sigma = 2 * sigma * sigma;
			kernel = new double[ size ];

			for ( int x = size / 2; x >= 0; --x )
			{
				final double val = Math.exp( -( x * x ) / two_sq_sigma );

				kernel[ size / 2 - x ] = val;
				kernel[ size / 2 + x ] = val;
			}
		}

		if ( normalize )
		{
			double sum = 0;
			for ( final double value : kernel )
				sum += value;

			for ( int i = 0; i < kernel.length; i++ )
				kernel[ i ] /= sum;
		}

		return kernel;
	}

	/**
     * Create a 1d-Gaussian kernel of appropriate size.
     *
     * @param sigma Standard deviation of the Gaussian kernel
     * @param normalize Normalize integral of the Gaussian kernel to 1 or not...
     *
     * @return float[] Gaussian kernel of appropriate size
     */
    final static public float[] createGaussianKernel(
    		final float sigma,
    		final boolean normalize )
	{
		float[] kernel;

		if ( sigma <= 0 )
		{
			kernel = new float[ 3 ];
			kernel[ 1 ] = 1;
		}
		else
		{
			final int size = Math.max( 3, ( int ) ( 2 * ( int )( 3 * sigma + 0.5 ) + 1 ) );

			final float two_sq_sigma = 2 * sigma * sigma;
			kernel = new float[ size ];

			for ( int x = size / 2; x >= 0; --x )
			{
				final float val = ( float ) Math.exp( -( float ) ( x * x ) / two_sq_sigma );

				kernel[ size / 2 - x ] = val;
				kernel[ size / 2 + x ] = val;
			}
		}

		if ( normalize )
		{
			float sum = 0;
			for ( final float value : kernel )
				sum += value;

			for ( int i = 0; i < kernel.length; i++ )
				kernel[ i ] /= sum;
		}

		return kernel;
	}

    /**
	 * Create a normalized 2d gaussian impulse with appropriate size with its
	 * center slightly moved away from the middle.
	 *
	 */
	final static public FloatArray2D createGaussianKernelOffset(
			final float sigma,
			final float offset_x,
			final float offset_y,
			final boolean normalize )
	{
		final FloatArray2D kernel;
		if ( sigma == 0 )
		{
			kernel = new FloatArray2D( 3, 3 );
			kernel.data[ 4 ] = 1;
		}
		else
		{
			final int size = Math.max( 3, ( int ) ( 2 * Math.round( 3 * sigma ) + 1 ) );
			final float two_sq_sigma = 2 * sigma * sigma;
			// float normalization_factor = 1.0/(float)M_PI/two_sq_sigma;
			kernel = new FloatArray2D( size, size );
			for ( int x = size - 1; x >= 0; --x )
			{
				final float fx = ( float ) ( x - size / 2 );
				for ( int y = size - 1; y >= 0; --y )
				{
					final float fy = ( float ) ( y - size / 2 );
					final float val = ( float ) ( Math.exp( -( Math.pow( fx - offset_x, 2 ) + Math.pow( fy - offset_y, 2 ) ) / two_sq_sigma ) );
					kernel.set( val, x, y );
				}
			}
		}
		if ( normalize )
		{
			float sum = 0;
			for ( final float value : kernel.data )
				sum += value;

			for ( int i = 0; i < kernel.data.length; i++ )
				kernel.data[ i ] /= sum;
		}
		return kernel;
	}

	/**
	 * Create a normalized 2d gaussian impulse with appropriate size with its
	 * center slightly moved away from the middle.
	 *
	 */
	final static public FloatArray2D createGaussianKernelOffset(
			final double sigma,
			final double offset_x,
			final double offset_y,
			final boolean normalize )
	{
		final FloatArray2D kernel;
		if ( sigma == 0 )
		{
			kernel = new FloatArray2D( 3, 3 );
			kernel.data[ 4 ] = 1;
		}
		else
		{
			final int size = Math.max( 3, ( int ) ( 2 * Math.round( 3 * sigma ) + 1 ) );
			final double two_sq_sigma = 2 * sigma * sigma;
			// float normalization_factor = 1.0/(float)M_PI/two_sq_sigma;
			kernel = new FloatArray2D( size, size );
			for ( int x = size - 1; x >= 0; --x )
			{
				final double fx = x - size / 2;
				for ( int y = size - 1; y >= 0; --y )
				{
					final double fy = y - size / 2;
					final double val = Math.exp( -( Math.pow( fx - offset_x, 2 ) + Math.pow( fy - offset_y, 2 ) ) / two_sq_sigma );
					kernel.set( ( float )val, x, y );
				}
			}
		}
		if ( normalize )
		{
			double sum = 0;
			for ( final float value : kernel.data )
				sum += value;

			for ( int i = 0; i < kernel.data.length; i++ )
				kernel.data[ i ] /= sum;
		}
		return kernel;
	}

	final public static FloatArray2D[] createGradients( final FloatArray2D array )
	{
		final FloatArray2D[] gradients = new FloatArray2D[ 2 ];
		gradients[ 0 ] = new FloatArray2D( array.width, array.height );
		gradients[ 1 ] = new FloatArray2D( array.width, array.height );

		for ( int y = 0; y < array.height; ++y )
		{
			final int[] ro = new int[ 3 ];
			ro[ 0 ] = array.width * Math.max( 0, y - 1 );
			ro[ 1 ] = array.width * y;
			ro[ 2 ] = array.width * Math.min( y + 1, array.height - 1 );
			for ( int x = 0; x < array.width; ++x )
			{
				// (L(x+1, y) - L(x-1, y)) / 2
				final float der_x = ( array.data[ ro[ 1 ] + Math.min( x + 1, array.width - 1 ) ] - array.data[ ro[ 1 ] + Math.max( 0, x - 1 ) ] ) / 2;

				// (L(x, y+1) - L(x, y-1)) / 2
				final float der_y = ( array.data[ ro[ 2 ] + x ] - array.data[ ro[ 0 ] + x ] ) / 2;

				// amplitude
				gradients[ 0 ].data[ ro[ 1 ] + x ] = ( float ) Math.sqrt( Math.pow( der_x, 2 ) + Math.pow( der_y, 2 ) );
				// orientation
				gradients[ 1 ].data[ ro[ 1 ] + x ] = ( float ) Math.atan2( der_y, der_x );
			}
		}
		return gradients;
	}

    /**
	 * In place enhance all values of a FloatArray to fill the given range.
	 *
	 * @param src
	 *            source
	 * @param scale
	 *            defines the range
	 */
    final static public void enhance( final FloatArray2D src, final float scale )
    {
    	float min = src.data[ 0 ];
    	float max = min;
    	for ( final float f : src.data )
    	{
    		if ( f < min ) min = f;
    		else if ( f > max ) max = f;
    	}
    	final float s = scale / ( max - min );
    	for ( int i = 0; i < src.data.length; ++i )
    		src.data[ i ] = s * ( src.data[ i ] - min );
    }

    /**
	 * Convolve an image with a horizontal and a vertical kernel.
	 *
	 * @param input the input image
	 * @param h horizontal kernel
	 * @param v vertical kernel
	 *
	 * @return convolved image
	 */
	final static public FloatArray2D convolveSeparable(
			final FloatArray2D input,
			final float[] h,
			final float[] v )
	{
		return convolveSeparable(input, h, v, null, null, null, null, 0);
	}

	/**
	 * Convolve an image with a horizontal and a vertical kernel and, while each output row is still in
	 * cache, also write the scaled differences to two other images:
	 * {@code dLower = ( output - lower ) * scale} and {@code dUpper = ( upper - output ) * scale}.
	 * These are the difference of Gaussian levels adjacent to a level of a
	 * {@link FloatArray2DScaleOctave}; computing them here saves a pass over three full-size images per
	 * level. Either pair may be null.
	 * <p>
	 * Both passes iterate the kernel taps in the outer loop and stream along a contiguous row in the
	 * inner loop, which C2 vectorizes. Every output pixel still accumulates its taps in ascending
	 * order from tap 0, so the result is bit-identical to the scalar reduction. The horizontally
	 * convolved rows are kept in a ring of {@code v.length} rows that stays in the L2 cache instead of
	 * a full-size temporary image that would be written to and read back from memory.
	 * </p>
	 */
	final static public FloatArray2D convolveSeparable(
			final FloatArray2D input,
			final float[] h,
			final float[] v,
			final FloatArray2D lower,
			final FloatArray2D dLower,
			final FloatArray2D upper,
			final FloatArray2D dUpper,
			final float scale
	) {
		final int w = input.width;
		final int height = input.height;
		final FloatArray2D output = new FloatArray2D(w, height);

		final int hl = h.length / 2;
		final int vl = v.length / 2;
		final int n = w - h.length + 1; // pixels per row that need no border handling

		// create lookup tables for coordinates outside the image range
		final int[] xb = new int[ h.length + hl - 1 ];
		final int[] xa = new int[ h.length + hl - 1 ];
		for ( int i = 0; i < xb.length; ++i )
		{
			xb[i] = Util.pingPong(i - hl, w);
			xa[i] = Util.pingPong(i + n, w);
		}

		final float[] in = input.data;
		final float[] out = output.data;
		final float[][] ring = new float[v.length][w];
		final float[] row = new float[w];
		final float[] acc = new float[w];
		final float[] diff = new float[w];

		int next = 0; // the next input row to be convolved horizontally
		for (int y = 0; y < height; ++y) {
			// output row y needs the input rows y - vl to y + vl (mirrored at the borders), so the ring holds all of them
			for (final int last = Math.min(height - 1, y + vl); next <= last; ++next)
				convolveRow(in, next * w, h, n, xb, xa, row, acc, ring[next % v.length]);
			convolveColumns(ring, v, y, height, acc);
			final int r = y * w;
			System.arraycopy(acc, 0, out, r, w);
			if (dLower != null) {
				System.arraycopy(lower.data, r, row, 0, w);
				scaledDifference(diff, acc, row, scale);
				System.arraycopy(diff, 0, dLower.data, r, w);
			}
			if (dUpper != null) {
				System.arraycopy(upper.data, r, row, 0, w);
				scaledDifference(diff, row, acc, scale);
				System.arraycopy(diff, 0, dUpper.data, r, w);
			}
		}

		return output;
	}

	/**
	 * Convolve the row of {@code in} starting at {@code r} with {@code h} into {@code t}. C2 (JDK 8 to
	 * 25) only vectorizes {@code acc[ x ] += k * src[ x ]} when both arrays are indexed by the same
	 * expression, so each shifted source row is copied into a scratch row first; the copy is a small
	 * fraction of the multiply-adds and stays in L1.
	 */
	private static void convolveRow(
			final float[] in,
			final int r,
			final float[] h,
			final int n,
			final int[] xb,
			final int[] xa,
			final float[] row,
			final float[] acc,
			final float[] t
	) {
		final int hl = h.length / 2;
		for (int xk = 0; xk < h.length; ++xk) {
			final float hk = h[xk];
			System.arraycopy(in, r + xk, row, 0, n);
			if (xk == 0)
				for (int x = 0; x < n; ++x)
					acc[x] = hk * row[x];
			else
				for (int x = 0; x < n; ++x)
					acc[x] += hk * row[x];
		}
		System.arraycopy(acc, 0, t, hl, n);
		for (int x = 0; x < hl; ++x) {
			float valb = 0;
			float vala = 0;
			for (int xk = 0; xk < h.length; ++xk) {
				valb += h[xk] * in[r + xb[x + xk]];
				vala += h[xk] * in[r + xa[x + xk]];
			}
			t[x] = valb;
			t[x + n + hl] = vala;
		}
	}

	/**
	 * {@code acc} = the rows of the ring for the input rows {@code y - vl ... y + vl} (mirrored at the
	 * borders) weighted by {@code v}. The ring rows are read in place, and four taps share one load and
	 * store of the accumulator; the taps are still added one after the other in ascending order.
	 */
	private static void convolveColumns(final float[][] ring, final float[] v, final int y, final int height, final float[] acc) {
		final int vl = v.length / 2;
		final int w = acc.length;
		final float v0 = v[0];
		final float[] t0 = ringRow(ring, y - vl, height);
		for (int x = 0; x < w; ++x)
			acc[x] = v0 * t0[x];
		int yk = 1;
		for (; yk + 3 < v.length; yk += 4) {
			final float v1 = v[yk], v2 = v[yk + 1], v3 = v[yk + 2], v4 = v[yk + 3];
			final float[] t1 = ringRow(ring, y - vl + yk, height), t2 = ringRow(ring, y - vl + yk + 1, height), t3 = ringRow(ring, y - vl + yk + 2, height), t4 = ringRow(ring, y - vl + yk + 3, height);
			for (int x = 0; x < w; ++x) {
				float a = acc[x];
				a += v1 * t1[x];
				a += v2 * t2[x];
				a += v3 * t3[x];
				a += v4 * t4[x];
				acc[x] = a;
			}
		}
		for (; yk < v.length; ++yk) {
			final float vk = v[yk];
			final float[] t = ringRow(ring, y - vl + yk, height);
			for (int x = 0; x < w; ++x)
				acc[x] += vk * t[x];
		}
	}

	/** the ring row holding input row {@code y}, mirrored into the image at the borders */
	private static float[] ringRow(final float[][] ring, final int y, final int height) {
		return ring[Util.pingPong(y, height) % ring.length];
	}

	/** {@code d = ( a - b ) * s} */
	private static void scaledDifference(final float[] d, final float[] a, final float[] b, final float s) {
		for (int x = 0; x < d.length; ++x)
			d[x] = (a[x] - b[x]) * s;
	}
}
