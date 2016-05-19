package io.github.mandar2812.PlasmaML.cdf.record;

import io.github.mandar2812.PlasmaML.cdf.Buf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Abstract superclass for a CDF Record object.
 * A Record represents one of the sequence of typed records
 * of which a CDF file is composed.
 *
 * @author   Mark Taylor
 * @since    18 Jun 2013
 */
public abstract class Record {

    private final RecordPlan plan_;
    private final String abbrev_;
    private final Logger logger_ = Logger.getLogger( Record.class.getName() );

    /**
     * Constructs a record with no known record type.
     *
     * @param  plan   basic record information
     * @param  abbrev   abreviated name for record type
     */
    protected Record( RecordPlan plan, String abbrev ) {
        plan_ = plan;
        abbrev_ = abbrev;
    }

    /**
     * Constructs a record with a known record type.
     *
     * @param  plan   basic record information
     * @param  abbrev   abreviated name for record type
     * @param  fixedType  record type asserted for this record
     */
    protected Record( RecordPlan plan, String abbrev, int fixedType ) {
        this( plan, abbrev );
        int planType = plan.getRecordType();

        // This really shouldn't happen.
        if ( planType != fixedType ) {
            throw new AssertionError( "Incorrect record type ("
                                    + planType + " != " + fixedType + ")" );
        }
    }

   
    /**
     * Returns the size of the record in bytes.
     *
     * @return  record size
     */
    public long getRecordSize() {
        return plan_.getRecordSize();
    }

    /**
     * Returns the type code identifying what kind of CDF record it is.
     *
     * @return   record type
     */
    public int getRecordType() {
        return plan_.getRecordType();
    }

    /**
     * Returns the buffer containing the record data.
     *
     * @return  buffer
     */
    public Buf getBuf() {
        return plan_.getBuf();
    }

    /**
     * Returns the abbreviated form of the record type for this record.
     *
     * @return  record type abbreviation
     */
    public String getRecordTypeAbbreviation() {
        return abbrev_;
    }

    /**
     * Returns the buffer offset of the first field in this record after
     * the record size and type values.
     *
     * @return   buffer offset for non-generic record content
     */
    public long getContentOffset() {
        return plan_.createContentPointer().get();
    }

    /**
     * Checks that an integer has a known fixed value.
     * If not, a warning may be emitted.
     * This performs an assertion-like function.
     * The actual value is returned as a convenience.
     *
     * @param  actualValue  value to test
     * @param  fixedValue  value to compare against
     * @return   <code>actualValue</code>
     */
    protected int checkIntValue( int actualValue, int fixedValue ) {
        if ( actualValue != fixedValue ) {
            warnFormat( "Unexpected fixed value " + actualValue + " != "
                       + fixedValue );
        }
        return actualValue;
    }

    /**
     * Checks that a pointer is positioned at the end of this record.
     * If not, a warning may be emitted.
     * This performs an assertion-like function.
     * This can be called by code which thinks it has read a whole record's
     * content to check that it's got the counting right.
     *
     * @param   ptr   pointer notionally positioned at end of record
     */
    protected void checkEndRecord( Pointer ptr ) {
        long readCount = plan_.getReadCount( ptr );
        long recSize = getRecordSize();
        if ( readCount != recSize ) {
            warnFormat( "Bytes read in record not equal to record size ("
                      + readCount + " != " + recSize + ")" );
        }
    }

    /**
     * Called by <code>check*</code> methods to issue a warning if the
     * check has failed.
     *
     * @param  msg  message to output
     */
    protected void warnFormat( String msg ) {
        assert false : msg;
        logger_.warning( msg );
    }

    /**
     * Reads a moderately-sized array of 4-byte big-endian integers.
     * Pointer position is moved on appropriately.
     * Not intended for potentially very large arrays.
     *
     * @param   buf  buffer
     * @param   ptr  pointer
     * @param   count  number of values to read
     * @return  <code>count</code>-element array of values
     */
    public static int[] readIntArray( Buf buf, Pointer ptr, int count )
            throws IOException {
        int[] array = new int[ count ];
        for ( int i = 0; i < count; i++ ) {
            array[ i ] = buf.readInt( ptr );
        }
        return array;
    }

    /**
     * Reads a moderately-sized offset 8-byte big-endian integers.
     * Pointer position is moved on appropriately.
     * Not intended for potentially very large arrays.
     *
     * @param   buf  buffer
     * @param   ptr  pointer
     * @param   count  number of values to read
     * @return  <code>count</code>-element array of values
     */
    public static long[] readOffsetArray( Buf buf, Pointer ptr, int count )
            throws IOException {
        long[] array = new long[ count ];
        for ( int i = 0; i < count; i++ ) {
            array[ i ] = buf.readOffset( ptr );
        }
        return array;
    }

    /**
     * Splits an ASCII string into 0x0A-terminated lines.
     *
     * @param   text  string containing ASCII characters
     * @return  array of lines split on linefeeds
     */
    public static String[] toLines( String text ) {
        List<String> lines = new ArrayList<String>();

        // Line ends in regexes are so inscrutable that use of String.split()
        // seems too much trouble.  See Goldfarb's First Law Of Text
        // Processing.
        int nc = text.length();
        StringBuilder sbuf = new StringBuilder( nc );
        for ( int i = 0; i < nc; i++ ) {
            char c = text.charAt( i );
            if ( c == 0x0a ) {
                lines.add( sbuf.toString() );
                sbuf.setLength( 0 );
            }
            else {
                sbuf.append( c );
            }
        }
        if ( sbuf.length() > 0 ) {
            lines.add( sbuf.toString() );
        }
        return lines.toArray( new String[ 0 ] );
    }

    /**
     * Indicates whether a given bit of a flags mask is set.
     *
     * @param  flags  flags mask
     * @param  ibit   bit index; 0 is the least significant
     */
    public static boolean hasBit( int flags, int ibit ) {
        return ( flags >> ibit ) % 2 == 1;
    }
}
