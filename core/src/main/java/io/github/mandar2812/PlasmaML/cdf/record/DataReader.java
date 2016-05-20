package io.github.mandar2812.PlasmaML.cdf.record;

import java.io.IOException;
import java.lang.reflect.Array;

import io.github.mandar2812.PlasmaML.cdf.Buf;
import io.github.mandar2812.PlasmaML.cdf.DataType;

/**
 * Reads items with a given data type from a buffer into an array.
 *
 * @author   Mark Taylor
 * @since    20 Jun 2013
 */
public class DataReader {

    private final DataType dataType_;
    private final int nelPerItem_;
    private final int nItem_;

    /**
     * Constructor.
     *
     * @param   dataType  data type
     * @param   nelPerItem  number of dataType elements per read item;
     *                      usually 1 except for character data
     * @param   nItem   number of items of given data type in the array,
     *                  for scalar records it will be 1
     */
    public DataReader( DataType dataType, int nelPerItem, int nItem ) {
        dataType_ = dataType;
        nelPerItem_ = nelPerItem;
        nItem_ = nItem;
    }

    /**
     * Creates a workspace array which can contain a value read for one record.
     * The return value will be an array of a primitive type or String.
     *
     * @return   workspace array for this reader
     */
    public Object createValueArray() {
        return Array.newInstance( dataType_.getArrayElementClass(),
                                  nItem_ * dataType_.getGroupSize() );
    }

    /**
     * Reads a value from a data buffer into a workspace array.
     *
     * @param  buf  data buffer
     * @param  offset  byte offset into buf of data start
     * @param  valueArray   object created by <code>createValueArray</code>
     *         into which results will be read
     */
    public void readValue(Buf buf, long offset, Object valueArray )
            throws IOException {
        dataType_.readValues( buf, offset, nelPerItem_, valueArray, nItem_ );
    }

    /**
     * Returns the size in bytes of one record as stored in the data buffer.
     *
     * @return   record size in bytes
     */
    public int getRecordSize() {
        return dataType_.getByteCount() * nelPerItem_ * nItem_;
    }
}
