package io.github.mandar2812.PlasmaML.cdf.util;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import io.github.mandar2812.PlasmaML.cdf.CdfField;
import io.github.mandar2812.PlasmaML.cdf.CdfReader;
import io.github.mandar2812.PlasmaML.cdf.Buf;
import io.github.mandar2812.PlasmaML.cdf.record.CdfDescriptorRecord;
import io.github.mandar2812.PlasmaML.cdf.record.GlobalDescriptorRecord;
import io.github.mandar2812.PlasmaML.cdf.OffsetField;
import io.github.mandar2812.PlasmaML.cdf.record.Record;
import io.github.mandar2812.PlasmaML.cdf.record.RecordFactory;

/**
 * Utility to dump the records of a CDF file, optionally with field values.
 * Intended to be used fro the command line via the <code>main</code> method.
 * The function is roughly comparable to the <code>cdfirsdump</code>
 * command in the CDF distribution.
 *
 * <p>The output can optionally be written in HTML format.
 * The point of this is so that field values which represent pointers
 * to records can be displayed as hyperlinks, which makes it very easy
 * to chase pointers around the CDF file in a web browser.
 *
 * @author   Mark Taylor
 * @since    21 Jun 2013
 */
public class CdfDump {

    private final CdfReader crdr_;
    private final PrintStream out_;
    private final boolean writeFields_;
    private final boolean html_;

    /**
     * Constructor.
     *
     * @param  crdr  CDF reader
     * @param  out   output stream for listing
     * @param  writeFields   true to write field data as well as record IDs
     * @param  html   true to write output in HTML format
     */
    public CdfDump( CdfReader crdr, PrintStream out, boolean writeFields,
                    boolean html ) {
        crdr_ = crdr;
        out_ = out;
        writeFields_ = writeFields;
        html_ = html;
    }

    /**
     * Does the work, writing output.
     */
    public void run() throws IOException {
        Buf buf = crdr_.getBuf();
        RecordFactory recFact = crdr_.getRecordFactory();
        long offset = 8;  // magic number
        long leng = buf.getLength();
        long eof = leng;
        CdfDescriptorRecord cdr = null;
        GlobalDescriptorRecord gdr = null;
        long gdroff = -1;
        if ( html_ ) {
            out_.println( "<html><body><pre>" );
        }
        for ( int ix = 0; offset < eof; ix++ ) {
            Record rec = recFact.createRecord( buf, offset );
            dumpRecord( ix, rec, offset );
            if ( cdr == null && rec instanceof CdfDescriptorRecord ) {
                cdr = (CdfDescriptorRecord) rec;
                gdroff = cdr.gdrOffset;
            }
            if ( offset == gdroff && rec instanceof GlobalDescriptorRecord ) {
                gdr = (GlobalDescriptorRecord) rec;
                eof = gdr.eof;
            }
            offset += rec.getRecordSize();
        }
        if ( html_ ) {
            out_.println( "<hr />" );
        }
        long extra = leng - eof;
        if ( extra > 0 ) {
            out_.println( " + " + extra + " bytes after final record" );
        }
        if ( html_ ) {
            out_.println( "</pre></body></html>" );
        }
    }

    /**
     * Writes infromation about a single record to the output.
     *
     * @param   index  record index
     * @param   rec   recor object
     * @param   offset  byte offset into the file of the record
     */
    private void dumpRecord( int index, Record rec, long offset ) {
        StringBuffer sbuf = new StringBuffer();
        if ( html_ ) {
            sbuf.append( "<hr /><strong>" );
        }
        sbuf.append( index )
            .append( ":\t" )
            .append( rec.getRecordTypeAbbreviation() )
            .append( "\t" )
            .append( rec.getRecordType() )
            .append( "\t" )
            .append( rec.getRecordSize() )
            .append( "\t" )
            .append( formatOffsetId( offset ) );
        if ( html_ ) {
            sbuf.append( "</strong>" );
        }
        out_.println( sbuf.toString() );

        // If required write the field values.  Rather than list them
        // for each record type, just obtain them by introspection.
        if ( writeFields_ ) {
            Field[] fields = rec.getClass().getFields();
            for ( int i = 0; i < fields.length; i++ ) {
                Field field = fields[ i ];
                if ( isCdfRecordField( field ) ) {
                    String name = field.getName();
                    Object value;
                    try {
                        value = field.get( rec );
                    }
                    catch ( IllegalAccessException e ) {
                        throw new RuntimeException( "Reflection error", e );
                    }
                    out_.println( formatFieldValue( name, value,
                                                    isOffsetField( field ) ) );
                }
            }
        }
    }

    /** 
     * Determines whether a given object field is a field of the CDF record.
     *
     * @param   field  field of java Record subclass
     * @return  true iff field represents a field of the corresponding CDF
     *          record type
     */
    private boolean isCdfRecordField( Field field ) {
        if ( field.getAnnotation( CdfField.class ) != null ) {
            int mods = field.getModifiers();
            assert Modifier.isFinal( mods )
                && Modifier.isPublic( mods )
                && ! Modifier.isStatic( mods );
            return true;
        }
        else {
            return false;
        }
    }

    /**
     * Determines whetehr a given object field represents a file offset.
     *
     * @param  field  field of java Record subclass
     * @return  true iff field represents a scalar or array file offset value
     */
    private boolean isOffsetField( Field field ) {
        return field.getAnnotation( OffsetField.class ) != null;
    }

    /**
     * Formats a field name/value pair for output.
     *
     * @param   name  field name
     * @param  value  field value
     */
    private String formatFieldValue( String name, Object value,
                                     boolean isOffset ) {
        StringBuffer sbuf = new StringBuffer();
        sbuf.append( spaces( 4 ) );
        sbuf.append( name )
            .append( ":" );
        sbuf.append( spaces( 28 - sbuf.length() ) );
        if ( value == null ) {
        }
        else if ( value.getClass().isArray() ) {
            int len = Array.getLength( value );
            if ( isOffset ) {
                assert value instanceof long[];
                long[] larray = (long[]) value;
                for ( int i = 0; i < len; i++ ) {
                    if ( i > 0 ) {
                        sbuf.append( ", " );
                    }
                    sbuf.append( formatOffsetRef( larray[ i ] ) );
                }
            }
            else {
                for ( int i = 0; i < len; i++ ) {
                    if ( i > 0 ) {
                        sbuf.append( ", " );
                    }
                    sbuf.append( Array.get( value, i ) );
                }
            }
        }
        else if ( isOffset ) {
            assert value instanceof Long;
            sbuf.append( formatOffsetRef( ((Long) value).longValue() ) );
        }
        else {
            sbuf.append( value.toString() );
        }
        return sbuf.toString();
    }

    /**
     * Format a value for output if it represents a possible target of
     * a pointer.
     *
     * @param  offset  pointer target value
     * @return   string for output
     */
    private String formatOffsetId( long offset ) {
        String txt = "0x" + Long.toHexString( offset );
        return html_ ? "<a name='" + txt + "'>" + txt + "</a>"
                     : txt;
    }

    /**
     * Format a value for output if it apparentl represents a pointer
     * to a particular file offset.
     *
     * @param  offset  target file offset
     * @return  string for output
     */
    private String formatOffsetRef( long offset ) {
        String txt = "0x" + Long.toHexString( offset );

        // Only format strictly positive values.  In some circumstances
        // -1 and 0 are used as special values indicating no reference exists.
        // The first record in any case starts at 0x8 (after the magic numbers)
        // so any such values can't be genuine offsets.
        return ( html_ && offset > 0L )
             ? "<a href='#" + txt + "'>" + txt + "</a>"
             : txt;
    }

    /**
     * Construct a padding string.
     *
     * @param   count   number of spaces
     * @return  string composed only of <code>count</code> spaces
     */
    static String spaces( int count ) {
        StringBuffer sbuf = new StringBuffer( count );
        for ( int i = 0; i < count; i++ ) {
            sbuf.append( ' ' );
        }
        return sbuf.toString();
    }

    /**
     * Does the work for the command line tool, handling arguments.
     * Sucess is indicated by the return value.
     *
     * @param  args   command-line arguments
     * @return   0 for success, non-zero for failure
     */
    public static int runMain( String[] args ) throws IOException {
        String usage = new StringBuffer()
           .append( "\n   Usage:" )
           .append( CdfDump.class.getName() )
           .append( " [-help]" )
           .append( " [-verbose]" )
           .append( " [-fields]" )
           .append( " [-html]" )
           .append( " <cdf-file>" )
           .append( "\n" )
           .toString();

        // Process arguments.
        List<String> argList = new ArrayList<String>( Arrays.asList( args ) );
        int verb = 0;
        File file = null;
        boolean writeFields = false;
        boolean html = false;
        for ( Iterator<String> it = argList.iterator(); it.hasNext(); ) {
            String arg = it.next();
            if ( arg.equals( "-html" ) ) {
                it.remove();
                html = true;
            }
            else if ( arg.startsWith( "-h" ) ) {
                it.remove();
                System.out.println( usage );
                return 0;
            }
            else if ( arg.equals( "-v" ) || arg.equals( "-verbose" ) ) {
                it.remove();
                verb++;
            }
            else if ( arg.equals( "+v" ) || arg.equals( "+verbose" ) ) {
                it.remove();
                verb--;
            }
            else if ( arg.startsWith( "-field" ) ) {
                it.remove();
                writeFields = true;
            }
            else if ( file == null ) {
                it.remove();
                file = new File( arg );
            }
        }

        // Validate arguments.
        if ( ! argList.isEmpty() ) {
            System.err.println( "Unused args: " + argList );
            System.err.println( usage );
            return 1;
        }
        if ( file == null ) {
            System.err.println( usage );
            return 1;
        }

        // Configure and run.
        LogUtil.setVerbosity( verb );
        new CdfDump( new CdfReader( file ), System.out, writeFields, html )
           .run();
        return 0;
    }

    /**
     * Main method.  Use -help for arguments.
     */
    public static void main( String[] args ) throws IOException {
        int status = runMain( args );
        if ( status != 0 ) {
            System.exit( status );
        }
    }
}
