package io.github.mandar2812.PlasmaML.cdf;

import java.io.IOException;
import java.lang.reflect.Array;

import io.github.mandar2812.PlasmaML.cdf.record.Pointer;

/**
 * Enumerates the data types supported by the CDF format.
 *
 * @author   Mark Taylor
 * @since    20 Jun 2013
 */
public abstract class DataType {

    private final String name_;
    private final int byteCount_;
    private final int groupSize_;
    private final Class<?> arrayElementClass_;
    private final Class<?> scalarClass_;
    private final Object dfltPadValueArray_;
    private boolean hasMultipleElementsPerItem_;

    public static final DataType INT1 = new Int1DataType( "INT1" );
    public static final DataType INT2 = new Int2DataType( "INT2" );
    public static final DataType INT4 = new Int4DataType( "INT4" );
    public static final DataType INT8 = new Int8DataType( "INT8" );
    public static final DataType UINT1 = new UInt1DataType( "UINT1" );
    public static final DataType UINT2 = new UInt2DataType( "UINT2" );
    public static final DataType UINT4 = new UInt4DataType( "UINT4" );
    public static final DataType REAL4 = new Real4DataType( "REAL4" );
    public static final DataType REAL8 = new Real8DataType( "REAL8" );
    public static final DataType CHAR = new CharDataType( "CHAR" );
    public static final DataType EPOCH16 = new Epoch16DataType( "EPOCH16" );
    public static final DataType BYTE = new Int1DataType( "BYTE" );
    public static final DataType FLOAT = new Real4DataType( "FLOAT" );
    public static final DataType DOUBLE = new Real8DataType( "DOUBLE" );
    public static final DataType EPOCH = new EpochDataType( "EPOCH" );
    public static final DataType TIME_TT2000 =
                                     new Tt2kDataType( "TIME_TT2000", -1 );
    public static final DataType UCHAR = new CharDataType( "UCHAR" );
    
    /**
     * Constructor.
     *
     * @param  name  type name
     * @param  byteCount  number of bytes to store one item
     * @param  groupSize  number of elements of type
     *                    <code>arrayElementClass</code> that are read
     *                    into the value array for a single item read
     * @param  arrayElementClass  component class of the value array
     * @param  scalarClass   object type returned by <code>getScalar</code>
     * @param  dfltPadValueArray  1-item array of arrayElementClass values
     *                            containing the default pad value for this type
     * @param  hasMultipleElementsPerItem  true iff a variable number of
     *             array elements may correspond to a single item
     */
    private DataType( String name, int byteCount, int groupSize,
                      Class<?> arrayElementClass, Class<?> scalarClass,
                      Object dfltPadValueArray,
                      boolean hasMultipleElementsPerItem ) {
        name_ = name;
        byteCount_ = byteCount;
        groupSize_ = groupSize;
        arrayElementClass_ = arrayElementClass;
        scalarClass_ = scalarClass;
        dfltPadValueArray_ = dfltPadValueArray;
        hasMultipleElementsPerItem_ = hasMultipleElementsPerItem;
    }

    /**
     * Constructor for a single-element-per-item type with a zero-like
     * pad value.
     *
     * @param  name  type name
     * @param  byteCount  number of bytes to store one item
     * @param  groupSize  number of elements of type
     *                    <code>arrayElementClass</code> that are read
     *                    into the value array for a single item read
     * @param  arrayElementClass  component class of the value array
     * @param  scalarClass   object type returned by <code>getScalar</code>
     */
    private DataType( String name, int byteCount, int groupSize,
                      Class<?> arrayElementClass, Class<?> scalarClass ) {
        this( name, byteCount, groupSize, arrayElementClass, scalarClass,
              Array.newInstance( arrayElementClass, groupSize ), false );
    }

    /**
     * Returns the name for this data type.
     *
     * @return  data type name
     */
    public String getName() {
        return name_;
    }

    /**
     * Returns the number of bytes used in a CDF to store a single item
     * of this type.
     *
     * @return  size in bytes
     */
    public int getByteCount() {
        return byteCount_;
    }

    /** 
     * Returns the element class of an array that this data type can
     * be read into.
     * In most cases this is a primitive type or String.
     *
     * @return   array raw value element class
     */
    public Class<?> getArrayElementClass() {
        return arrayElementClass_;
    }

    /**
     * Returns the type of objects obtained by the <code>getScalar</code>
     * method.
     *
     * @return   scalar type associated with this data type
     */
    public Class<?> getScalarClass() {
        return scalarClass_;
    }

    /**
     * Number of elements of type arrayElementClass that are read into
     * valueArray for a single item read.
     * This is usually 1, but not, for instance, for EPOCH16.
     *
     * @return   number of array elements per item
     */
    public int getGroupSize() {
        return groupSize_;
    }

    /**
     * Returns the index into a value array which corresponds to the
     * <code>item</code>'th element.
     *
     * @return   <code>itemIndex</code> * <code>groupSize</code>
     */
    public int getArrayIndex( int itemIndex ) {
        return groupSize_ * itemIndex;
    }

    /**
     * True if this type may turn a variable number of elements from the
     * value array into a single read item.  This is usually false,
     * but true for character types, which turn into strings.
     *
     * @return  true  iff type may have multiple elements per read item
     */
    public boolean hasMultipleElementsPerItem() {
        return hasMultipleElementsPerItem_;
    }

    /**
     * Returns an array of array-class values containing a single item
     * with the default pad value for this type.
     *
     * @return  default raw pad value array
     * @see  "Section 2.3.20 of CDF User's Guide"
     */
    public Object getDefaultPadValueArray() {
        return dfltPadValueArray_;
    }

    /**
     * Reads data of this data type from a buffer into an appropriately
     * typed value array.
     *
     * @param   buf  data buffer
     * @param   offset  byte offset into buffer at which data starts
     * @param   nelPerItem  number of elements per item;
     *                      usually 1, but may not be for strings
     * @param   valueArray  array to receive result data
     * @param   count  number of items to read
     */
    public abstract void readValues( Buf buf, long offset, int nelPerItem,
                                     Object valueArray, int count )
            throws IOException;

    /** 
     * Reads a single item from an array which has previously been
     * populated by {@link #readValues readValues}.
     * The class of the returned value is that returned by
     * {@link #getScalarClass}.
     *
     * <p>The <code>arrayIndex</code> argument is the index into the 
     * array object, not necessarily the item index -
     * see the {@link #getArrayIndex getArrayIndex} method.
     *
     * @param   valueArray  array filled with data for this data type
     * @param  arrayIndex  index into array at which the item to read is found
     * @return  scalar representation of object at position <code>index</code>
     *          in <code>valueArray</code>
     */
    public abstract Object getScalar( Object valueArray, int arrayIndex );

    /**
     * Provides a string view of a scalar value obtained for this data type.
     *
     * @param  value   value returned by <code>getScalar</code>
     * @return   string representation
     */
    public String formatScalarValue( Object value ) {
        return value == null ? "" : value.toString();
    }

    /**
     * Provides a string view of an item obtained from an array value
     * of this data type.
     * <p>The <code>arrayIndex</code> argument is the index into the 
     * array object, not necessarily the item index -
     * see the {@link #getArrayIndex getArrayIndex} method.
     *
     * @param   array  array value populated by <code>readValues</code>
     * @param   arrayIndex  index into array
     * @return  string representation
     */
    public String formatArrayValue( Object array, int arrayIndex ) {
        Object value = Array.get( array, arrayIndex );
        return value == null ? "" : value.toString();
    }

    @Override
    public String toString() {
        return name_;
    }

    /**
     * Returns a DataType corresponding to a CDF data type code,
     * possibly customised for a particular CDF file.
     *
     * <p>Currently, this returns the same as <code>getDataType(int)</code>,
     * except for TIME_TT2000 columns, in which case the last known leap
     * second may be taken into account.
     *
     * @param  dataType  dataType field of AEDR or VDR
     * @param  cdfInfo   specifics of CDF file
     * @return   data type object
     */
    public static DataType getDataType( int dataType, CdfInfo cdfInfo )
            throws CdfFormatException {
        DataType type = getDataType( dataType );
        return type == TIME_TT2000
             ? new Tt2kDataType( type.getName(),
                                 cdfInfo.getLeapSecondLastUpdated() )
             : type;
    }

    /**
     * Returns the DataType object corresponding to a CDF data type code.
     *
     * @param  dataType  dataType field of AEDR or VDR
     * @return   data type object
     */
    public static DataType getDataType( int dataType )
            throws CdfFormatException {
        switch ( dataType ) {
            case  1: return INT1;
            case  2: return INT2;
            case  4: return INT4;
            case  8: return INT8;
            case 11: return UINT1;
            case 12: return UINT2;
            case 14: return UINT4;
            case 41: return BYTE;
            case 21: return REAL4;
            case 22: return REAL8;
            case 44: return FLOAT;
            case 45: return DOUBLE;
            case 31: return EPOCH;
            case 32: return EPOCH16;
            case 33: return TIME_TT2000;
            case 51: return CHAR;
            case 52: return UCHAR;
            default:
                throw new CdfFormatException( "Unknown data type " + dataType );
        }
    }

    /**
     * DataType for signed 1-byte integer.
     */
    private static final class Int1DataType extends DataType {
        Int1DataType( String name ) {
            super( name, 1, 1, byte.class, Byte.class );
        }
        public void readValues( Buf buf, long offset, int nelPerItem,
                                Object array, int n ) throws IOException {
            buf.readDataBytes( offset, n, (byte[]) array );
        }
        public Object getScalar( Object array, int index ) {
            return new Byte( ((byte[]) array)[ index ] );
        }
    }

    /**
     * DataType for signed 2-byte integer.
     */
    private static final class Int2DataType extends DataType {
        Int2DataType( String name ) {
            super( name, 2, 1, short.class, Short.class );
        }
        public void readValues( Buf buf, long offset, int nelPerItem,
                                Object array, int n ) throws IOException {
            buf.readDataShorts( offset, n, (short[]) array );
        }
        public Object getScalar( Object array, int index ) {
            return new Short( ((short[]) array)[ index ] );
        }
    }

    /**
     * DataType for signed 4-byte integer.
     */
    private static final class Int4DataType extends DataType {
        Int4DataType( String name ) {
            super( name, 4, 1, int.class, Integer.class );
        }
        public void readValues( Buf buf, long offset, int nelPerItem,
                                Object array, int n ) throws IOException {
            buf.readDataInts( offset, n, (int[]) array );
        }
        public Object getScalar( Object array, int index ) {
            return new Integer( ((int[]) array)[ index ] );
        }
    }

    /**
     * DataType for signed 8-byte integer.
     */
    private static class Int8DataType extends DataType {
        Int8DataType( String name ) {
            super( name, 8, 1, long.class, Long.class );
        }
        public void readValues( Buf buf, long offset, int nelPerItem,
                                Object array, int n ) throws IOException {
            buf.readDataLongs( offset, n, (long[]) array );
        }
        public Object getScalar( Object array, int index ) {
            return new Long( ((long[]) array)[ index ] );
        }
    }

    /**
     * DataType for unsigned 1-byte integer.
     * Output values are 2-byte signed integers because of the difficulty
     * of handling unsigned integers in java.
     */
    private static class UInt1DataType extends DataType {
        UInt1DataType( String name ) {
            super( name, 1, 1, short.class, Short.class );
        }
        public void readValues( Buf buf, long offset, int nelPerItem,
                                Object array, int n ) throws IOException {
            Pointer ptr = new Pointer( offset );
            short[] sarray = (short[]) array;
            for ( int i = 0; i < n; i++ ) {
                sarray[ i ] = (short) buf.readUnsignedByte( ptr );
            }
        }
        public Object getScalar( Object array, int index ) {
            return new Short( ((short[]) array)[ index ] );
        }
    }

    /**
     * DataType for unsigned 2-byte integer.
     * Output vaules are 4-byte signed integers because of the diffculty
     * of handling unsigned integers in java.
     */
    private static class UInt2DataType extends DataType {
        UInt2DataType( String name ) {
            super( name, 2, 1, int.class, Integer.class );
        }
        public void readValues( Buf buf, long offset, int nelPerItem,
                                Object array, int n ) throws IOException {
            Pointer ptr = new Pointer( offset );
            int[] iarray = (int[]) array;
            boolean bigend = buf.isBigendian();
            for ( int i = 0; i < n; i++ ) {
                int b0 = buf.readUnsignedByte( ptr );
                int b1 = buf.readUnsignedByte( ptr );
                iarray[ i ] = bigend ? b1 | ( b0 << 8 )
                                     : b0 | ( b1 << 8 );
            }
        }
        public Object getScalar( Object array, int index ) {
            return new Integer( ((int[]) array)[ index ] );
        }
    }

    /** 
     * DataType for unsigned 4-byte integer.
     * Output values are 8-byte signed integers because of the difficulty
     * of handling unsigned integers in java.
     */
    private static class UInt4DataType extends DataType {
        UInt4DataType( String name ) {
            super( name, 4, 1, long.class, Long.class );
        }
        public void readValues( Buf buf, long offset, int nelPerItem,
                                Object array, int n ) throws IOException {
            Pointer ptr = new Pointer( offset );
            long[] larray = (long[]) array;
            boolean bigend = buf.isBigendian();
            for ( int i = 0; i < n; i++ ) {
                long b0 = buf.readUnsignedByte( ptr );
                long b1 = buf.readUnsignedByte( ptr );
                long b2 = buf.readUnsignedByte( ptr );
                long b3 = buf.readUnsignedByte( ptr );
                larray[ i ] = bigend
                            ? b3 | ( b2 << 8 ) | ( b1 << 16 ) | ( b0 << 24 )
                            : b0 | ( b1 << 8 ) | ( b2 << 16 ) | ( b3 << 24 );
            }
        }
        public Object getScalar( Object array, int index ) {
            return new Long( ((long[]) array )[ index ] );
        }
    }

    /**
     * DataType for 4-byte floating point.
     */
    private static class Real4DataType extends DataType {
        Real4DataType( String name ) {
            super( name, 4, 1, float.class, Float.class );
        }
        public void readValues( Buf buf, long offset, int nelPerItem,
                                Object array, int n ) throws IOException {
            buf.readDataFloats( offset, n, (float[]) array );
        }
        public Object getScalar( Object array, int index ) {
            return new Float( ((float[]) array)[ index ] );
        }
    }

    /**
     * DataType for 8-byte floating point.
     */
    private static class Real8DataType extends DataType {
        Real8DataType( String name ) {
            super( name, 8, 1, double.class, Double.class );
        }
        public void readValues( Buf buf, long offset, int nelPerItem,
                                Object array, int n ) throws IOException {
            buf.readDataDoubles( offset, n, (double[]) array );
        }
        public Object getScalar( Object array, int index ) {
            return new Double( ((double[]) array)[ index ] );
        }
    }

    /**
     * DataType for TIME_TT2000.  May be qualified by last known leap second.
     */
    private static class Tt2kDataType extends Int8DataType {
        final int leapSecondLastUpdated_;
        final EpochFormatter formatter_;
        final long[] dfltPad_ = new long[] { Long.MIN_VALUE + 1 };
        Tt2kDataType( String name, int leapSecondLastUpdated ) {
            super( name );
            leapSecondLastUpdated_ = leapSecondLastUpdated;
            formatter_ = new EpochFormatter( leapSecondLastUpdated );
        }
        @Override
        public Object getDefaultPadValueArray() {
            return dfltPad_;
        }
        @Override
        public String formatScalarValue( Object value ) {
            synchronized ( formatter_ ) {
                return formatter_
                      .formatTimeTt2000( ((Long) value).longValue() );
            }
        }
        @Override
        public String formatArrayValue( Object array, int index ) {
            synchronized ( formatter_ ) {
                return formatter_
                      .formatTimeTt2000( ((long[]) array)[ index ] );
            }
        }
        @Override
        public int hashCode() {
            int code = 392552;
            code = 23 * code + leapSecondLastUpdated_;
            return code;
        }
        @Override
        public boolean equals( Object o ) {
            if ( o instanceof Tt2kDataType ) {
                Tt2kDataType other = (Tt2kDataType) o;
                return this.leapSecondLastUpdated_ ==
                       other.leapSecondLastUpdated_;
            }
            else {
                return false;
            }
        }
    }

    /**
     * DataType for 1-byte character.
     * Output is as numElem-character String.
     */
    private static class CharDataType extends DataType {
        CharDataType( String name ) {
            super( name, 1, 1, String.class, String.class,
                   new String[] { null }, true );
        }
        public void readValues( Buf buf, long offset, int nelPerItem,
                                Object array, int n ) throws IOException {
            String[] sarray = (String[]) array;
            byte[] cbuf = new byte[ nelPerItem * n ];
            buf.readDataBytes( offset, nelPerItem * n, cbuf );
            for ( int i = 0; i < n; i++ ) {
                @SuppressWarnings("deprecation")
                String s = new String( cbuf, i * nelPerItem, nelPerItem );
                sarray[ i ] = s;
            }
        }
        public Object getScalar( Object array, int index ) {
            return ((String[]) array)[ index ];
        }
    }

    /**
     * DataType for 8-byte floating point epoch.
     */
    private static class EpochDataType extends Real8DataType {
        private final EpochFormatter formatter_ = new EpochFormatter();
        EpochDataType( String name ) {
            super( name );
        }
        @Override
        public String formatScalarValue( Object value ) {
            synchronized ( formatter_ ) {
                return formatter_.formatEpoch( ((Double) value).doubleValue() );
            }
        }
        @Override
        public String formatArrayValue( Object array, int index ) {
            synchronized ( formatter_ ) {
                return formatter_.formatEpoch( ((double[]) array)[ index ] );
            }
        }
    }

    /**
     * DataType for 16-byte (2*double) epoch.
     * Output is as a 2-element array of doubles.
     */
    private static class Epoch16DataType extends DataType {
        private final EpochFormatter formatter_ = new EpochFormatter();
        Epoch16DataType( String name ) {
            super( name, 16, 2, double.class, double[].class );
        }
        public void readValues( Buf buf, long offset, int nelPerItem,
                                Object array, int n ) throws IOException {
            buf.readDataDoubles( offset, n * 2, (double[]) array );
        }
        public Object getScalar( Object array, int index ) {
            double[] darray = (double[]) array;
            return new double[] { darray[ index ], darray[ index + 1 ] };
        }
        @Override
        public String formatScalarValue( Object value ) {
            double[] v2 = (double[]) value;
            synchronized ( formatter_ ) {
                return formatter_.formatEpoch16( v2[ 0 ], v2[ 1 ] );
            }
        }
        @Override
        public String formatArrayValue( Object array, int index ) {
            double[] darray = (double[]) array;
            synchronized ( formatter_ ) {
                return formatter_.formatEpoch16( darray[ index ],
                                                 darray[ index + 1 ] );
            }
        }
    }
}
