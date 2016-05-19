package io.github.mandar2812.PlasmaML.cdf;

import io.github.mandar2812.PlasmaML.cdf.record.Pointer;

import java.io.IOException;
import java.io.InputStream;

/**
 * Represents a sequence of bytes along with operations to read
 * primitive values from it.
 * This interface abstracts away implementation details such as storage
 * mechanism, data encoding and pointer length.
 * It is capable of dealing with 64-bit lengths and offsets.
 * All of the <code>read*</code> methods are safe for use from multiple
 * threads concurrently.
 *
 * @author   Mark Taylor
 * @since    18 Jun 2013
 */
public interface Buf {

    /**
     * Returns the extent of this buf in bytes.
     *
     * @return  buffer length
     */
    long getLength();

    /**
     * Reads a single byte from the pointer position,
     * returning a value in the range 0..255.
     * Pointer position is moved on appropriately.
     *
     * @param  ptr   pointer
     * @return   byte value
     */
    int readUnsignedByte( Pointer ptr ) throws IOException;

    /**
     * Reads a signed big-endian 4-byte integer from the pointer position.
     * Pointer position is moved on appropriately.
     *
     * @param  ptr  pointer
     * @return  integer value
     */
    int readInt( Pointer ptr ) throws IOException;

    /**
     * Reads a file offset or size from the pointer position.
     * This is a signed big-endian integer,
     * occupying either 4 or 8 bytes according
     * to the return value of {@link #isBit64}.
     * Pointer position is moved on appropriately.
     *
     * @return  buffer size or offset value
     */
    long readOffset( Pointer ptr ) throws IOException;

    /**
     * Reads a fixed number of bytes interpreting them as ASCII characters
     * and returns the result as a string.
     * If a character 0x00 appears before <code>nbyte</code> bytes have
     * been read, it is taken as the end of the string.
     * Pointer position is moved on appropriately.
     *
     * @param   ptr    pointer
     * @param  nbyte   maximum number of bytes in string
     * @return  ASCII string
     */
    String readAsciiString( Pointer ptr, int nbyte ) throws IOException;

    /**
     * Sets the 64bit-ness of this buf.
     * This determines whether {@link #readOffset readOffset} reads
     * 4- or 8-byte values.
     *
     * <p>This method should be called before the <code>readOffset</code>
     * method is invoked.
     *
     * @param  isBit64  true for 8-byte offsets, false for 4-byte offsets
     */
    void setBit64( boolean isBit64 );

    /**
     * Determines the 64bit-ness of this buf.
     * This determines whether {@link #readOffset readOffset} reads
     * 4- or 8-byte values.
     *
     * @return  true for 8-byte offsets, false for 4-byte offsets
     */
    boolean isBit64();

    /**
     * Sets the encoding for reading numeric values as performed by the
     * <code>readData*</code> methods.
     *
     * <p>As currently specified, there are only two possibiliies,
     * Big-Endian and Little-Endian.  Interface and implementation would
     * need to be reworked somewhat to accommodate the 
     * (presumably, rarely seen in this day and age)
     * D_FLOAT and G_FLOAT encodings supported by the CDF standard.
     *
     * <p>This method should be called before any of the <code>readData*</code>
     * methods are invoked.
     *
     * @param  isBigendian  true for big-endian, false for little-endian
     */
    void setEncoding( boolean isBigendian );

    /**
     * Determines the data encoding of this buf.
     *
     * @return  true for big-endian, false for little-endian
     */
    boolean isBigendian();

    /**
     * Reads a sequence of byte values from this buf into an array.
     *
     * @param  offset  position sequence start in this buffer in bytes
     * @param  count   number of byte values to read
     * @param  array   array to receive values, starting at array element 0
     */
    void readDataBytes( long offset, int count, byte[] array )
            throws IOException;

    /**
     * Reads a sequence of short values from this buf into an array.
     *
     * @param  offset  position sequence start in this buffer in bytes
     * @param  count   number of short values to read
     * @param  array   array to receive values, starting at array element 0
     */
    void readDataShorts( long offset, int count, short[] array )
            throws IOException;

    /**
     * Reads a sequence of int values from this buf into an array.
     *
     * @param  offset  position sequence start in this buffer in bytes
     * @param  count   number of int values to read
     * @param  array   array to receive values, starting at array element 0
     */
    void readDataInts( long offset, int count, int[] array )
            throws IOException;

    /**
     * Reads a sequence of long integer values from this buf into an array.
     *
     * @param  offset  position sequence start in this buffer in bytes
     * @param  count   number of long values to read
     * @param  array   array to receive values, starting at array element 0
     */
    void readDataLongs( long offset, int count, long[] array )
            throws IOException;

    /**
     * Reads a sequence of float values from this buf into an array.
     *
     * @param  offset  position sequence start in this buffer in bytes
     * @param  count   number of float values to read
     * @param  array   array to receive values, starting at array element 0
     */
    void readDataFloats( long offset, int count, float[] array )
            throws IOException;

    /**
     * Reads a sequence of double values from this buf into an array.
     *
     * @param  offset  position sequence start in this buffer in bytes
     * @param  count   number of double values to read
     * @param  array   array to receive values, starting at array element 0
     */
    void readDataDoubles( long offset, int count, double[] array )
            throws IOException;

    /**
     * Returns an input stream consisting of all the bytes in this buf
     * starting from the given offset.
     *
     * @param  offset  position of first byte in buf that will appear in
     *                 the returned stream
     * @return  input stream
     */
    InputStream createInputStream( long offset );

    /**
     * Creates a new Buf of a given length populated from a given input stream.
     * The new buf object must have the same data encoding and 64bit-ness
     * as this one.
     *
     * @param  count  size of new buffer in bytes
     * @param  in  input stream capable of supplying
     *             (at least) <code>count</code> bytes
     * @return  new buffer of length <code>count</code> filled with bytes
     *          from <code>in</code>
     */
    Buf fillNewBuf( long count, InputStream in ) throws IOException;
}
