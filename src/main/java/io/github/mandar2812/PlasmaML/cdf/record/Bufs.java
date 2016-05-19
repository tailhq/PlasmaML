package io.github.mandar2812.PlasmaML.cdf.record;

import io.github.mandar2812.PlasmaML.cdf.Buf;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.logging.Logger;

/**
 * Factory and utility methods for use with Bufs.
 *
 * @author   Mark Taylor
 * @since    21 Jun 2013
 */
public class Bufs {

    /** Preferred maximum size for a bank buffer.  */
    private static final int BANK_SIZE = 1 << 30;
    private static Logger logger_ = Logger.getLogger( Bufs.class.getName() );

    /**
     * Private constructor prevents instantiation.
     */
    private Bufs() {
    }

    /**
     * Creates a buf based on a single NIO buffer.
     *
     * @param   byteBuffer  NIO buffer containing data
     * @param   isBit64  64bit-ness of buf
     * @param   isBigendian   true for big-endian data, false for little-endian
     */
    public static Buf createBuf(ByteBuffer byteBuffer,
                                boolean isBit64, boolean isBigendian ) {
        return new SimpleNioBuf( byteBuffer, isBit64, isBigendian );
    }

    /**
     * Creates a buf based on a sequence of NIO buffers.
     *
     * @param   byteBuffers  array of NIO buffers containing data
     * @param   isBit64  64bit-ness of buf
     * @param   isBigendian   true for big-endian data, false for little-endian
     */
    public static Buf createBuf( ByteBuffer[] byteBuffers,
                                 boolean isBit64, boolean isBigendian ) {
        return byteBuffers.length == 1
             ? createBuf( byteBuffers[ 0 ], isBit64, isBigendian )
             : BankBuf.createMultiBankBuf( byteBuffers, isBit64, isBigendian );
    }

    /**
     * Creates a buf based on a file.
     *
     * @param  file  file containing data
     * @param   isBit64  64bit-ness of buf
     * @param   isBigendian   true for big-endian data, false for little-endian
     */
    public static Buf createBuf( File file,
                                 boolean isBit64, boolean isBigendian )
            throws IOException {
        FileChannel channel = new FileInputStream( file ).getChannel();
        long leng = file.length();
        if ( leng <= Integer.MAX_VALUE ) {
            int ileng = (int) leng;
            ByteBuffer bbuf =
                channel.map( FileChannel.MapMode.READ_ONLY, 0, ileng );
            return createBuf( bbuf, isBit64, isBigendian );
        }
        else {
            return BankBuf.createMultiBankBuf( channel, leng, BANK_SIZE,
                                               isBit64, isBigendian );
        }
    }

    /**
     * Decompresses part of an input Buf into an output Buf.
     *
     * @param  compression  compression format 
     * @param  inBuf   buffer containing input compressed data
     * @param  inOffset   offset into <code>inBuf</code> at which the
     *                    compressed data starts
     * @param   outSize  byte count of the uncompressed data
     * @return   new buffer of size <code>outSize</code> containing
     *           uncompressed data
     */
    public static Buf uncompress( Compression compression, Buf inBuf,
                                  long inOffset, long outSize )
            throws IOException {
        logger_.config( "Uncompressing CDF data to new " + outSize
                      + "-byte buffer" );
        InputStream uin =
             compression
            .uncompressStream( new BufferedInputStream(
                                   inBuf.createInputStream( inOffset ) ) );
        Buf ubuf = inBuf.fillNewBuf( outSize, uin );
        uin.close();
        return ubuf;
    }

    /**
     * Utility method to acquire the data from an NIO buffer in the form
     * of an InputStream.
     *
     * @param   bbuf  NIO buffer
     * @return  stream
     */
    public static InputStream createByteBufferInputStream( ByteBuffer bbuf ) {
        return new ByteBufferInputStream( bbuf );
    }

    // Utility methods to read arrays of data from buffers.
    // These essentially provide bulk absolute NIO buffer read operations;
    // The NIO Buffer classes themselves only provide relative read operations
    // for bulk reads.
    //
    // We work differently according to whether we are in fact reading
    // single value or multiple values.  This is because NIO Buffer
    // classes have absolute read methods for scalar reads, but only
    // relative read methods for array reads (i.e. you need to position
    // a pointer and then do the read).  For thread safety we need to
    // synchronize in that case to make sure somebody else doesn't
    // reposition before the read takes place.
    //
    // For the array reads, we also recast the ByteBuffer to a Buffer of
    // the appropriate type for the data being read.
    // 
    // Both these steps are taken on the assumption that the bulk reads
    // are more efficient than multiple byte reads perhaps followed by
    // bit manipulation where required.  The NIO javadocs suggest that
    // assumption is true, but I haven't tested it.  Doing it the other
    // way would avoid the need for synchronization.

    /**
     * Utility method to read a fixed length ASCII string from an NIO buffer.
     * If a character 0x00 is encountered before the end of the byte sequence,
     * it is considered to terminate the string.
     *
     * @param  bbuf  NIO buffer
     * @param  ioff  offset into buffer of start of string
     * @param  nbyte  number of bytes in string
     */
    static String readAsciiString( ByteBuffer bbuf, int ioff, int nbyte ) {
        byte[] abuf = new byte[ nbyte ];
        synchronized ( bbuf ) {
            bbuf.position( ioff );
            bbuf.get( abuf, 0, nbyte );
        }
        StringBuffer sbuf = new StringBuffer( nbyte );
        for ( int i = 0; i < nbyte; i++ ) {
            byte b = abuf[ i ];
            if ( b == 0 ) {
                break;
            }
            else {
                sbuf.append( (char) b );
            }
        }
        return sbuf.toString();
    }

    /**
     * Utility method to read an array of byte values from an NIO buffer
     * into an array.
     *
     * @param  bbuf  buffer
     * @param  ioff  offset into bbuf of data start
     * @param  count  number of values to read
     * @param  a    array into which values will be read, starting at element 0
     */
    static void readBytes( ByteBuffer bbuf, int ioff, int count, byte[] a ) {
        if ( count == 1 ) {
            a[ 0 ] = bbuf.get( ioff );
        }
        else {
            synchronized ( bbuf ) {
                bbuf.position( ioff );
                bbuf.get( a, 0, count );
            }
        }
    }

    /**
     * Utility method to read an array of short values from an NIO buffer
     * into an array.
     *
     * @param  bbuf  buffer
     * @param  ioff  offset into bbuf of data start
     * @param  count  number of values to read
     * @param  a    array into which values will be read, starting at element 0
     */
    static void readShorts( ByteBuffer bbuf, int ioff, int count, short[] a ) {
        if ( count == 1 ) {
            a[ 0 ] = bbuf.getShort( ioff );
        }
        else {
            synchronized ( bbuf ) {
                bbuf.position( ioff );
                bbuf.asShortBuffer().get( a, 0, count );
            }
        }
    }

    /**
     * Utility method to read an array of int values from an NIO buffer
     * into an array.
     *
     * @param  bbuf  buffer
     * @param  ioff  offset into bbuf of data start
     * @param  count  number of values to read
     * @param  a    array into which values will be read, starting at element 0
     */
    static void readInts( ByteBuffer bbuf, int ioff, int count, int[] a ) {
        if ( count == 1 ) {
            a[ 0 ] = bbuf.getInt( ioff );
        }
        else {
            synchronized ( bbuf ) {
                bbuf.position( ioff );
                bbuf.asIntBuffer().get( a, 0, count );
            }
        }
    }

    /**
     * Utility method to read an array of long values from an NIO buffer
     * into an array.
     *
     * @param  bbuf  buffer
     * @param  ioff  offset into bbuf of data start
     * @param  count  number of values to read
     * @param  a    array into which values will be read, starting at element 0
     */
    static void readLongs( ByteBuffer bbuf, int ioff, int count, long[] a ) {
        if ( count == 1 ) {
            a[ 0 ] = bbuf.getLong( ioff );
        }
        else {
            synchronized ( bbuf ) {
                bbuf.position( ioff );
                bbuf.asLongBuffer().get( a, 0, count );
            }
        }
    }

    /**
     * Utility method to read an array of float values from an NIO buffer
     * into an array.
     *
     * @param  bbuf  buffer
     * @param  ioff  offset into bbuf of data start
     * @param  count  number of values to read
     * @param  a    array into which values will be read, starting at element 0
     */
    static void readFloats( ByteBuffer bbuf, int ioff, int count, float[] a ) {
        if ( count == 1 ) {
            a[ 0 ] = bbuf.getFloat( ioff );
        }
        else {
            synchronized ( bbuf ) {
                bbuf.position( ioff );
                bbuf.asFloatBuffer().get( a, 0, count );
            }
        }
    }

    /**
     * Utility method to read an array of double values from an NIO buffer
     * into an array.
     *
     * @param  bbuf  buffer
     * @param  ioff  offset into bbuf of data start
     * @param  count  number of values to read
     * @param  a    array into which values will be read, starting at element 0
     */
    static void readDoubles( ByteBuffer bbuf, int ioff, int count,
                             double[] a ) {
        if ( count == 1 ) {
            a[ 0 ] = bbuf.getDouble( ioff );
        }
        else {
            synchronized ( bbuf ) {
                bbuf.position( ioff );
                bbuf.asDoubleBuffer().get( a, 0, count );
            }
        }
    }

    /**
     * Input stream that reads from an NIO buffer.
     * You'd think there was an implementation of this in the J2SE somewhere,
     * but I can't see one.
     */
    private static class ByteBufferInputStream extends InputStream {
        private final ByteBuffer bbuf_;

        /**
         * Constructor.
         *
         * @param  bbuf  NIO buffer supplying data
         */
        ByteBufferInputStream( ByteBuffer bbuf ) {
            bbuf_ = bbuf;
        }

        @Override
        public int read() {
            return bbuf_.remaining() > 0 ? bbuf_.get() : -1;
        }              
            
        @Override 
        public int read( byte[] b ) {
            return read( b, 0, b.length );
        }

        @Override
        public int read( byte[] b, int off, int len ) {
            if ( len == 0 ) {
                return 0;
            }
            int remain = bbuf_.remaining();
            if ( remain == 0 ) {
                return -1;
            }
            else {
                int nr = Math.min( remain, len );
                bbuf_.get( b, off, nr );
                return nr;
            }
        }

        @Override
        public boolean markSupported() {
            return true;
        }

        @Override
        public void mark( int readLimit ) {
            bbuf_.mark();
        }

        @Override
        public void reset() {
            bbuf_.reset();
        }

        @Override
        public long skip( long n ) {
            int nsk = (int) Math.min( n, bbuf_.remaining() );
            bbuf_.position( bbuf_.position() + nsk );
            return nsk;
        }

        @Override
        public int available() {
            return bbuf_.remaining();
        }
    }
}
