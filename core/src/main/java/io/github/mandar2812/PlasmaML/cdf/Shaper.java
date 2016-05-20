package io.github.mandar2812.PlasmaML.cdf;

import java.lang.reflect.Array;
import java.util.Arrays;

/**
 * Takes care of turning raw variable record values into shaped
 * record values.  The raw values are those stored in the CDF data stream,
 * and the shaped ones are those notionally corresponding to record values.
 *
 * @author   Mark Taylor
 * @since    20 Jun 2013
 */
public abstract class Shaper {

    private final int[] dimSizes_;
    private final boolean[] dimVarys_;

    /**
     * Constructor.
     *
     * @param   dimSizes  dimensionality of shaped array
     * @param   dimVarys  for each dimension, true for varying, false for fixed
     */
    protected Shaper( int[] dimSizes, boolean[] dimVarys ) {
        dimSizes_ = dimSizes;
        dimVarys_ = dimVarys;
    }

    /**
     * Returns the number of array elements in the raw value array.
     *
     * @return  raw value array size
     */
    public abstract int getRawItemCount();

    /**
     * Returns the number of array elements in the shaped value array.
     *
     * @return  shaped value array size
     */
    public abstract int getShapedItemCount();

    /** 
     * Returns the dimensions of the notional array.
     *
     * @return   dimension sizes array
     */
    public int[] getDimSizes() {
        return dimSizes_;
    }

    /**
     * Returns the dimension variances of the array.
     *
     * @return   for each dimension, true if the data varies, false if fixed
     */
    public boolean[] getDimVarys() {
        return dimVarys_;
    }

    /**
     * Returns the data type of the result of the {@link #shape shape} method.
     *
     * @return  shaped value class
     */
    public abstract Class<?> getShapeClass();

    /**
     * Takes a raw value array and turns it into an object of
     * the notional shape for this shaper.
     * The returned object is new; it is not rawValue.
     *
     * @param   rawValue  input raw value array
     * @return  rowMajor  required majority for result;
     *                    true for row major, false for column major
     */
    public abstract Object shape( Object rawValue, boolean rowMajor );

    /**
     * Returns the index into the raw value array at which the value for
     * the given element of the notional array can be found.
     *
     * @param   coords  coordinate array, same length as dimensionality
     * @return  index into raw value array
     */
    public abstract int getArrayIndex( int[] coords );

    /**
     * Returns an appropriate shaper instance.
     *
     * @param   dataType  data type
     * @param   dimSizes  dimensions of notional shaped array
     * @param   dimVarys  variances of shaped array
     * @param   rowMajor  majority of raw data array;
     *                    true for row major, false for column major
     */
    public static Shaper createShaper( DataType dataType,
                                       int[] dimSizes, boolean[] dimVarys,
                                       boolean rowMajor ) {
        int rawItemCount = 1;
        int shapedItemCount = 1;
        int nDimVary = 0;
        int ndim = dimSizes.length;
        for ( int idim = 0; idim < dimSizes.length; idim++ ) {
            int dimSize = dimSizes[ idim ];
            shapedItemCount *= dimSize;
            if ( dimVarys[ idim ] ) {
                nDimVary++;
                rawItemCount *= dimSize;
            }
        }
        if ( shapedItemCount == 1 ) {
            return new ScalarShaper( dataType );
        }
        else if ( ndim == 1  && nDimVary == 1 ) {
            assert Arrays.equals( dimVarys, new boolean[] { true } ); 
            assert Arrays.equals( dimSizes, new int[] { rawItemCount } );
            return new VectorShaper( dataType, rawItemCount );
        }
        else if ( nDimVary == ndim ) {
            return new SimpleArrayShaper( dataType, dimSizes, rowMajor );
        }
        else {
            return new GeneralShaper( dataType, dimSizes, dimVarys, rowMajor );
        }
    }

    /**
     * Shaper implementation for scalar values.  Easy.
     */
    private static class ScalarShaper extends Shaper {
        private final DataType dataType_;

        /**
         * Constructor.
         *
         * @param  dataType  data type
         */
        ScalarShaper( DataType dataType ) {
            super( new int[ 0 ], new boolean[ 0 ] );
            dataType_ = dataType;
        }
        public int getRawItemCount() {
            return 1;
        }
        public int getShapedItemCount() {
            return 1;
        }
        public Class<?> getShapeClass() {
            return dataType_.getScalarClass();
        }
        public Object shape( Object rawValue, boolean rowMajor ) {
            return dataType_.getScalar( rawValue, 0 );
        }
        public int getArrayIndex( int[] coords ) {
            for ( int i = 0; i < coords.length; i++ ) {
                if ( coords[ i ] != 0 ) {
                    throw new IllegalArgumentException( "Out of bounds" );
                }
            }
            return 0;
        }
    }

    /**
     * Shaper implementation for 1-dimensional arrays with true dimension
     * variance along the single dimension.
     * No need to worry about majority, since the question doesn't arise
     * in one dimension.
     */
    private static class VectorShaper extends Shaper {
        private final DataType dataType_;
        private final int itemCount_;
        private final int step_;
        private final Class<?> shapeClass_;

        /**
         * Constructor.
         *
         * @param  dataType  data type
         * @param  itemCount   number of elements in raw and shaped arrays
         */
        VectorShaper( DataType dataType, int itemCount ) {
            super( new int[] { itemCount }, new boolean[] { true } );
            dataType_ = dataType;
            itemCount_ = itemCount;
            step_ = dataType.getGroupSize();
            shapeClass_ = getArrayClass( dataType.getArrayElementClass() );
        }
        public int getRawItemCount() {
            return itemCount_;
        }
        public int getShapedItemCount() {
            return itemCount_;
        }
        public Class<?> getShapeClass() {
            return shapeClass_;
        }
        public Object shape( Object rawValue, boolean rowMajor ) {
            Object out = Array.newInstance( dataType_.getArrayElementClass(),
                                            itemCount_ );

            // Contract requires that we return a new object.
            System.arraycopy( rawValue, 0, out, 0, itemCount_ );
            return out;
        }
        public int getArrayIndex( int[] coords ) {
            return coords[ 0 ] * step_;
        }
    }

    /**
     * Shaper implementation that can deal with multiple dimensions,
     * majority switching, and dimension variances,
     */
    private static class GeneralShaper extends Shaper {

        private final DataType dataType_;
        private final int[] dimSizes_;
        private final boolean rowMajor_;
        private final int ndim_;
        private final int rawItemCount_;
        private final int shapedItemCount_;
        private final int[] strides_;
        private final int itemSize_;
        private final Class<?> shapeClass_;
        
        /**
         * Constructor.
         *
         * @param   dataType  data type
         * @param   dimSizes  dimensionality of shaped array
         * @param   dimVarys  variances of shaped array
         * @param   rowMajor  majority of raw data array;
         *                    true for row major, false for column major
         */
        GeneralShaper( DataType dataType, int[] dimSizes, boolean[] dimVarys,
                       boolean rowMajor ) {
            super( dimSizes, dimVarys );
            dataType_ = dataType;
            dimSizes_ = dimSizes;
            rowMajor_ = rowMajor;
            ndim_ = dimSizes.length;

            int rawItemCount = 1;
            int shapedItemCount = 1;
            int nDimVary = 0;
            int ndim = dimSizes.length;
            strides_ = new int[ ndim_ ];
            for ( int idim = 0; idim < ndim_; idim++ ) {
                int jdim = rowMajor ? ndim_ - idim - 1 : idim;
                int dimSize = dimSizes[ jdim ];
                shapedItemCount *= dimSize;
                if ( dimVarys[ jdim ] ) {
                    nDimVary++;
                    strides_[ jdim ] = rawItemCount;
                    rawItemCount *= dimSize;
                }
            }
            rawItemCount_ = rawItemCount;
            shapedItemCount_ = shapedItemCount;
            itemSize_ = dataType_.getGroupSize();
            shapeClass_ = getArrayClass( dataType.getArrayElementClass() );
        }

        public int getRawItemCount() {
            return rawItemCount_;
        }

        public int getShapedItemCount() {
            return shapedItemCount_;
        }

        public int getArrayIndex( int[] coords ) {
            int index = 0;
            for ( int idim = 0; idim < ndim_; idim++ ) {
                index += coords[ idim ] * strides_[ idim ];
            }
            return index * itemSize_;
        }

        public Class<?> getShapeClass() {
            return shapeClass_;
        }

        public Object shape( Object rawValue, boolean rowMajor ) {
            Object out = Array.newInstance( dataType_.getArrayElementClass(),
                                            shapedItemCount_ * itemSize_ );
            int[] coords = new int[ ndim_ ];
            Arrays.fill( coords, -1 );
            for ( int ix = 0; ix < shapedItemCount_; ix++ ) {
                for ( int idim = 0; idim < ndim_; idim++ ) {
                    int jdim = rowMajor ? ndim_ - idim - 1 : idim;
                    coords[ jdim ] = ( coords[ jdim ] + 1 ) % dimSizes_[ jdim ];
                    if ( coords[ jdim ] != 0 ) {
                        break;
                    }
                }
                System.arraycopy( rawValue, getArrayIndex( coords ),
                                  out, ix * itemSize_, itemSize_ );
            }
            return out;
        }
    }

    /**
     * Shaper implementation that can deal with multiple dimensions and
     * majority switching, but not false dimension variances.
     */
    private static class SimpleArrayShaper extends GeneralShaper {

        private final DataType dataType_;
        private final boolean rowMajor_;

        /**
         * Constructor.
         *
         * @param   dataType  data type
         * @param   dimSizes  dimensionality of shaped array
         * @param   rowMajor  majority of raw data array;
         *                    true for row major, false for column major
         */
        public SimpleArrayShaper( DataType dataType, int[] dimSizes,
                                  boolean rowMajor ) {
            super( dataType, dimSizes, trueArray( dimSizes.length ),
                   rowMajor );
            dataType_ = dataType;
            rowMajor_ = rowMajor;
        }

        public Object shape( Object rawValue, boolean rowMajor ) {
            if ( rowMajor == rowMajor_ ) {
                int count = Array.getLength( rawValue );
                Object out =
                    Array.newInstance( dataType_.getArrayElementClass(),
                                       count );
                System.arraycopy( rawValue, 0, out, 0, count );
                return out;
            }
            else {
                // Probably there's a more efficient way to do this -
                // it's an n-dimensional generalisation of transposing
                // a matrix (though don't forget to keep units of
                // groupSize intact).
                return super.shape( rawValue, rowMajor );
            }
        }

        /**
         * Utility method that returns a boolean array of a given size
         * populated with true values.
         *
         * @param  n  size
         * @return   n-element array filled with true
         */
        private static boolean[] trueArray( int n ) {
            boolean[] a = new boolean[ n ];
            Arrays.fill( a, true );
            return a;
        }
    }

    /**
     * Returns the array class corresponding to a given scalar class.
     *
     * @param  elementClass  scalar class
     * @return   array class
     */
    private static Class<?> getArrayClass( Class elementClass ) {
        return Array.newInstance( elementClass, 0 ).getClass();
    }
}
