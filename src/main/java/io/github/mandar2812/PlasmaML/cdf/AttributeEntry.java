package io.github.mandar2812.PlasmaML.cdf;

/**
 * Represents an entry in a global or variable attribute.
 *
 * @author   Mark Taylor
 * @since    28 Jun 2013
 */
public class AttributeEntry {

    private final DataType dataType_;
    private final Object rawValue_;
    private final int nitem_;

    /**
     * Constructor.
     *
     * @param  dataType  data type
     * @param  rawValue  array object storing original representation
     *                   of the object in the CDF (array of primitives or
     *                   Strings)
     * @param  nitem     number of items represented by the array
     */
    public AttributeEntry( DataType dataType, Object rawValue, int nitem ) {
        dataType_ = dataType;
        rawValue_ = rawValue;
        nitem_ = nitem;
    }

    /**
     * Returns the data type of this entry.
     *
     * @return  data type
     */
    public DataType getDataType() {
        return dataType_;
    }

    /**
     * Returns the array object storing the original representation
     * of the object in the CDF.  This is either an array of either
     * primitives or Strings.
     *
     * @return  raw array value
     */
    public Object getRawValue() {
        return rawValue_;
    }

    /**
     * Returns the value of this entry as a convenient object.
     * If the item count is 1 it's the same as <code>getItem(0)</code>,
     * and if the item count is &gt;1 it's the same as the raw value.
     *
     * @return  shaped entry value
     */
    public Object getShapedValue() {
        if ( nitem_ == 0 ) {
            return null;
        }
        else if ( nitem_ == 1 ) {
            return dataType_.getScalar( rawValue_, 0 );
        }
        else {
            return rawValue_;
        }
    }

    /**
     * Returns the number of items in this entry.
     *
     * @return  item count
     */
    public int getItemCount() {
        return nitem_;
    }

    /**
     * Returns an object representing one of the items in this entry.
     * If the raw array is a primitive, the result is a wrapper object.
     *
     * @param  itemIndex  item index
     * @return  value of item
     */
    public Object getItem( int itemIndex ) {
        return dataType_.getScalar( rawValue_,
                                    dataType_.getArrayIndex( itemIndex ) );
    }

    /**
     * Formats the value of this entry as a string.
     */
    @Override
    public String toString() {
        if ( rawValue_ == null || nitem_ == 0 ) {
            return "";
        }
        else {
            StringBuffer sbuf = new StringBuffer();
            for ( int i = 0; i < nitem_; i++ ) {
                if ( i > 0 ) {
                    sbuf.append( ", " );
                }
                sbuf.append( dataType_
                            .formatArrayValue( rawValue_,
                                               dataType_.getArrayIndex( i ) ) );
            }
            return sbuf.toString();
        }
    }
}
