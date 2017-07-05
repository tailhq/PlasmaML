package io.github.mandar2812.PlasmaML.cdf;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.util.logging.Logger;

import io.github.mandar2812.PlasmaML.cdf.record.Bufs;
import io.github.mandar2812.PlasmaML.cdf.record.CdfDescriptorRecord;
import io.github.mandar2812.PlasmaML.cdf.record.CompressedCdfRecord;
import io.github.mandar2812.PlasmaML.cdf.record.CompressedParametersRecord;
import io.github.mandar2812.PlasmaML.cdf.record.Compression;
import io.github.mandar2812.PlasmaML.cdf.record.Pointer;
import io.github.mandar2812.PlasmaML.cdf.record.Record;
import io.github.mandar2812.PlasmaML.cdf.record.RecordFactory;

/**
 * Examines a CDF file and provides methods to access its records.
 *
 * <p>Constructing an instance of this class reads enough of a file
 * to identify it as a CDF and work out how to access its records.
 * Most of the actual contents are only read from the data buffer
 * as required.
 * Although only the magic numbers and CDR are read during construction,
 * in the case of a file-compressed CDF the whole thing is uncompressed,
 * so it could still be an expensive operation.
 *
 * <p>For low-level access to the CDF internal records, use the
 * {@link #getCdr} method to get the CdfDescriptorRecord and use that
 * in conjunction with knowledge of the internal format of CDF files
 * as a starting point to chase pointers around the file constructing
 * other records.  When you have a pointer to another record, you can
 * use the record factory got from {@link #getRecordFactory} to turn
 * it into a typed Record object.
 *
 * @author   Mark Taylor
 * @since    19 Jun 2013
 */
public class CdfReader {

    private final CdfDescriptorRecord cdr_;
    private final Buf buf_;
    private final RecordFactory recordFactory_;

    private static final Logger logger_ =
        Logger.getLogger( CdfReader.class.getName() );

    /** 
     * Constructs a CdfReader from a buffer containing its byte data.
     *
     * @param   buf  buffer containing CDF file
     */
    public CdfReader( Buf buf ) throws IOException {
        Pointer ptr = new Pointer( 0 );

        // Read the CDF magic number bytes.
        int magic1 = buf.readInt( ptr );
        int magic2 = buf.readInt( ptr );
        int offsetRec0 = (int) ptr.get();

        // Work out from that what variant (if any) of the CDF format
        // this file implements.
        CdfVariant variant = decodeMagic( magic1, magic2 );
        if ( variant == null ) {
            String msg = new StringBuffer()
                .append( "Unrecognised magic numbers: " )
                .append( "0x" )
                .append( Integer.toHexString( magic1 ) )
                .append( ", " )
                .append( "0x" )
                .append( Integer.toHexString( magic2 ) )
                .toString();
            throw new CdfFormatException( msg );
        }
        logger_.config( "CDF magic number for " + variant.label_ );
        logger_.config( "Whole file compression: " + variant.compressed_ );

        // The length of the pointers and sizes used in CDF files are
        // dependent on the CDF file format version.
        // Notify the buffer which regime is in force for this file.
        // Note that no operations for which this makes a difference have
        // yet taken place.
        buf.setBit64( variant.bit64_ );

        // The lengths of some fields differ according to CDF version.
        // Construct a record factory that does it right.
        recordFactory_ = new RecordFactory( variant.nameLeng_ );

        // Read the CDF Descriptor Record.  This may be the first record,
        // or it may be in a compressed form along with the rest of
        // the internal records.
        if ( variant.compressed_ ) {

            // Work out compression type and location of compressed data.
            CompressedCdfRecord ccr =
                recordFactory_.createRecord( buf, offsetRec0,
                                             CompressedCdfRecord.class );
            CompressedParametersRecord cpr =
                recordFactory_.createRecord( buf, ccr.cprOffset,
                                             CompressedParametersRecord.class );
            final Compression compress =
                Compression.getCompression( cpr.cType );

            // Uncompress the compressed data into a new buffer.
            // The compressed data is the data record of the CCR.
            // When uncompressed it can be treated just like the whole of
            // an uncompressed CDF file, except that it doesn't have the
            // magic numbers (8 bytes) prepended to it.
            // Note however that any file offsets recorded within the file
            // are given as if the magic numbers are present - this is not
            // very clear from the Internal Format Description document,
            // but it appears to be the case from reverse engineering
            // whole-file compressed files.  To work round this, we hack
            // the compression to prepend a dummy 8-byte block to the
            // uncompressed stream it provides.
            final int prepad = offsetRec0;
            assert prepad == 8;
            Compression padCompress =
                    new Compression( "Padded " + compress.getName() ) {
                public InputStream uncompressStream( InputStream in )
                        throws IOException {
                    InputStream in1 =
                        new ByteArrayInputStream( new byte[ prepad ] );
                    InputStream in2 = compress.uncompressStream( in );
                    return new SequenceInputStream( in1, in2 );
                }
            };
            buf = Bufs.uncompress( padCompress, buf, ccr.getDataOffset(),
                                   ccr.uSize + prepad );
        }
        cdr_ = recordFactory_.createRecord( buf, offsetRec0,
                                            CdfDescriptorRecord.class );

        // Interrogate CDR for required information.
        boolean isSingleFile = Record.hasBit( cdr_.flags, 1 );
        if ( ! isSingleFile ) {
            throw new CdfFormatException( "Multi-file CDFs not supported" );
        }
        NumericEncoding encoding =
            NumericEncoding.getEncoding( cdr_.encoding );
        Boolean bigEndian = encoding.isBigendian();
        if ( bigEndian == null ) {
            throw new CdfFormatException( "Unsupported encoding " + encoding );
        }
        buf.setEncoding( bigEndian.booleanValue() );
        buf_ = buf;
    }

    /**
     * Constructs a CdfReader from a readable file containing its byte data.
     *
     * @param  file  CDF file
     */
    public CdfReader( File file ) throws IOException {
        this( Bufs.createBuf( file, true, true ) );
    }

    /**
     * Returns the buffer containing the uncompressed record stream for
     * this reader's CDF file.
     * This will be the buffer originally submitted at construction time
     * only if the CDF does not use whole-file compression.
     *
     * @return   buffer containing CDF records
     */
    public Buf getBuf() {
        return buf_;
    }

    /** 
     * Returns a RecordFactory that can be applied to this reader's Buf 
     * to construct CDF Record objects.
     *
     * @return  record factory
     */
    public RecordFactory getRecordFactory() {
        return recordFactory_;
    }

    /**
     * Returns the CDF Descriptor Record object for this reader's CDF.
     *
     * @return  CDF Descriptor Record
     */
    public CdfDescriptorRecord getCdr() {
        return cdr_;
    }

    /**
     * Examines a byte array to see if it looks like the start of a CDF file.
     *
     * @param   intro  byte array, at least 8 bytes if available
     * @return  true iff the first 8 bytes of <code>intro</code> are
     *          a CDF magic number
     */
    public static boolean isMagic( byte[] intro ) {
        if ( intro.length < 8 ) {
            return false;
        }
        return decodeMagic( readInt( intro, 0 ), readInt( intro, 4 ) ) != null;
    }

    /**
     * Reads an 4-byte big-endian integer from a byte array.
     *
     * @param  b  byte array
     * @param  ioff   index into <code>b</code> of integer start
     * @return   int value
     */
    private static int readInt( byte[] b, int ioff ) {
        return ( b[ ioff++ ] & 0xff ) << 24
             | ( b[ ioff++ ] & 0xff ) << 16
             | ( b[ ioff++ ] & 0xff ) <<  8
             | ( b[ ioff++ ] & 0xff ) <<  0;
    }

    /**
     * Interprets two integer values as the magic number sequence at the
     * start of a CDF file, and returns an object encoding the information
     * about CDF encoding specifics.
     *
     * @param   magic1  big-endian int at CDF file offset 0x00
     * @param   magic2  big-endian int at CDF file offset 0x04
     * @return  object describing CDF encoding specifics,
     *          or null if this is not a recognised CDF magic number
     */
    private static CdfVariant decodeMagic( int magic1, int magic2 ) {
        final String label;
        final boolean bit64;
        final int nameLeng;
        final boolean compressed;
        if ( magic1 == 0xcdf30001 ) {  // version 3.0 - 3.4 (3.*?)
            label = "V3";
            bit64 = true;
            nameLeng = 256;
            if ( magic2 == 0x0000ffff ) {
                compressed = false;
            }
            else if ( magic2 == 0xcccc0001 ) {
                compressed = true;
            }
            else {
                return null;
            }
        }
        else if ( magic1 == 0xcdf26002 ) {  // version 2.6/2.7
            label = "V2.6/2.7";
            bit64 = false;
            nameLeng = 64;
            if ( magic2 == 0x0000ffff ) {
                compressed = false;
            }
            else if ( magic2 == 0xcccc0001 ) {
                compressed = true;
            }
            else {
                return null;
            }
        }
        else if ( magic1 == 0x0000ffff ) { // pre-version 2.6
            label = "pre-V2.6";
            bit64 = false;
            nameLeng = 64; // true as far as I know
            if ( magic2 == 0x0000ffff ) {
                compressed = false;
            }
            else {
                return null;
            }
        }
        else {
            return null;
        }
        return new CdfVariant( label, bit64, nameLeng, compressed );
    }

    /**
     * Encapsulates CDF encoding details as determined from the magic number.
     */
    private static class CdfVariant {
        final String label_;
        final boolean bit64_;
        final int nameLeng_;
        final boolean compressed_;

        /**
         * Constructor.
         *
         * @param  label  short string indicating CDF format version number
         * @param  bit64  true for 8-bit pointers, false for 4-bit pointers
         * @param  nameLeng  number of bytes used for attribute and variable
         *                   names
         * @param  compressed true iff the CDF file uses whole-file compression
         */
        CdfVariant( String label, boolean bit64, int nameLeng,
                    boolean compressed ) {
            label_ = label;
            bit64_ = bit64;
            nameLeng_ = nameLeng;
            compressed_ = compressed;
        }
    }
}
