package io.github.mandar2812.PlasmaML.cdf.record;

import java.io.IOException;
import java.io.InputStream;
import java.util.zip.GZIPInputStream;

import io.github.mandar2812.PlasmaML.cdf.BitExpandInputStream;
import io.github.mandar2812.PlasmaML.cdf.CdfFormatException;
import io.github.mandar2812.PlasmaML.cdf.RunLengthInputStream;

/**
 * Defines a data compression type supported for compressing CDF data.
 *
 * @author   Mark Taylor
 * @since    19 Jun 2013
 */
public abstract class Compression {

    /** No compression. */
    public static final Compression NONE = new Compression( "NONE" ) {
        public InputStream uncompressStream( InputStream in ) {
            return in;
        }
    };

    /** Run length encoding. */
    public static final Compression RLE = new Compression( "RLE" ) {
        public InputStream uncompressStream( InputStream in )
                throws IOException {
            return new RunLengthInputStream( in, (byte) 0 );
        }
    };

    /** Huffman encoding. */
    public static final Compression HUFF = new Compression( "HUFF" ) {
        public InputStream uncompressStream( InputStream in )
                throws IOException {
            return new BitExpandInputStream.HuffmanInputStream( in );
        }
    };

    /** Adaptive Huffman encoding. */
    public static final Compression AHUFF = new Compression( "AHUFF" ) {
        public InputStream uncompressStream( InputStream in )
                throws IOException {
            return new BitExpandInputStream.AdaptiveHuffmanInputStream( in );
        }
    };

    /** Gzip compression. */
    public static final Compression GZIP = new Compression( "GZIP" ) {
        public InputStream uncompressStream( InputStream in )
                throws IOException {
            return new GZIPInputStream( in );
        }
    };

    private final String name_;

    /**
     * Constructor.
     *
     * @param   name   compression format name
     */
    protected Compression( String name ) {
        name_ = name;
    }

    /**
     * Turns a stream containing compressed data into a stream containing
     * uncompressed data.
     *
     * @param  in  compressed input stream
     * @return  uncompressed input stream
     */
    public abstract InputStream uncompressStream( InputStream in )
            throws IOException;

    /**
     * Returns this compression format's name.
     *
     * @return  name
     */
    public String getName() {
        return name_;
    }

    /**
     * Returns a Compression object corresponding to a given compression code.
     *
     * @param  cType  compression code, as taken from the CPR cType field
     * @return  compression object
     * @throws CdfFormatException if the compression type is unknown
     */
    public static Compression getCompression( int cType )
            throws CdfFormatException {

        // The mapping is missing from the CDF Internal Format Description
        // document, but cdf.h says:
        //    #define NO_COMPRESSION                  0L
        //    #define RLE_COMPRESSION                 1L
        //    #define HUFF_COMPRESSION                2L
        //    #define AHUFF_COMPRESSION               3L
        //    #define GZIP_COMPRESSION                5L
        switch ( cType ) {
            case 0: return NONE;
            case 1: return RLE;
            case 2: return HUFF;
            case 3: return AHUFF;
            case 5: return GZIP;
            default:
                throw new CdfFormatException( "Unknown compression format "
                                            + "cType=" + cType );
        }
    }
}
