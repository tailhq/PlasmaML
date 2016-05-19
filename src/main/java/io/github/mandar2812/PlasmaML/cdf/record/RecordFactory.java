package io.github.mandar2812.PlasmaML.cdf.record;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import io.github.mandar2812.PlasmaML.cdf.Buf;
import io.github.mandar2812.PlasmaML.cdf.CdfFormatException;

/**
 * Turns bytes in a buffer into typed and populated CDF records.
 *
 * @author   Mark Taylor
 * @since    18 Jun 2013
 */
public class RecordFactory {

    private final Map<Integer,TypedRecordFactory> factoryMap_;
    private final Logger logger_ =
        Logger.getLogger( RecordFactory.class.getName() );

    /**
     * Constructor.
     *
     * @param  nameLeng   number of bytes in variable and attribute names;
     *                    appears to be 64 for pre-v3 and 256 for v3
     */
    public RecordFactory( int nameLeng ) {
        factoryMap_ = createFactoryMap( nameLeng );
    }

    /**
     * Creates a Record object from a given position in a buffer.
     * The returned object will be an instance of one of the
     * Record subclasses as appropriate for its type.
     *
     * @param  buf  byte buffer
     * @param  offset  start of record in buf
     * @return  record
     */
    public Record createRecord(Buf buf, long offset ) throws IOException {
        Pointer ptr = new Pointer( offset );
        long recSize = buf.readOffset( ptr );
        int recType = buf.readInt( ptr );
        RecordPlan plan = new RecordPlan( offset, recSize, recType, buf );
        TypedRecordFactory tfact = factoryMap_.get( recType );
        if ( tfact == null ) {
            throw new CdfFormatException( "Unknown record type " + recType );
        }
        else {
            Record rec = tfact.createRecord( plan );
            String msg = new StringBuffer()
               .append( "CDF Record:\t" )
               .append( "0x" )
               .append( Long.toHexString( offset ) )
               .append( "\t+" )
               .append( recSize )
               .append( "\t" )
               .append( rec.getRecordTypeAbbreviation() )
               .toString();
            logger_.config( msg );
            return rec;
        }
    }

    /** 
     * Creates a Record object with a known type from a given position in
     * a buffer.  This simply calls the untyped <code>getRecord</code>
     * method, and attempts to cast the result, throwing a
     * CdfFormatException if it has the wrong type.
     *
     * @param  buf  byte buffer
     * @param  offset  start of record in buf
     * @param  clazz   record class asserted for the result
     * @return  record
     * @throws  CdfFormatException  if the record found there turns out
     *          not to be of type <code>clazz</code>
     */
    public <R extends Record> R createRecord( Buf buf, long offset,
                                              Class<R> clazz )
            throws IOException {
        Record rec = createRecord( buf, offset );
        if ( clazz.isInstance( rec ) ) {
            return clazz.cast( rec );
        }
        else {
            String msg = new StringBuffer()
                .append( "Unexpected record type at " )
                .append( "0x" )
                .append( Long.toHexString( offset ) )
                .append( "; got " )
                .append( rec.getClass().getName() )
                .append( " not " )
                .append( clazz.getName() )
                .toString();
            throw new CdfFormatException( msg );
        }
    }

    /**
     * Sets up a mapping from CDF RecordType codes to factories for the
     * record types in question.
     *
     * @return   map of record type to record factory
     */
    private static Map<Integer,TypedRecordFactory>
            createFactoryMap( final int nameLeng ) {
        Map<Integer,TypedRecordFactory> map =
            new HashMap<Integer,TypedRecordFactory>();
        map.put( 1, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new CdfDescriptorRecord( plan );
            }
        } );
        map.put( 2, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new GlobalDescriptorRecord( plan );
            }
        } );
        map.put( 4, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new AttributeDescriptorRecord( plan, nameLeng );
            }
        } );
        map.put( 5, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new AttributeEntryDescriptorRecord.GrVariant( plan );
            }
        } );
        map.put( 9, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new AttributeEntryDescriptorRecord.ZVariant( plan );
            }
        } );
        map.put( 3, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new VariableDescriptorRecord.RVariant( plan, nameLeng );
            }
        } );
        map.put( 8, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new VariableDescriptorRecord.ZVariant( plan, nameLeng );
            }
        } );
        map.put( 6, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new VariableIndexRecord( plan );
            }
        } );
        map.put( 7, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new VariableValuesRecord( plan );
            }
        } );
        map.put( 10, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new CompressedCdfRecord( plan );
            }
        } );
        map.put( 11, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new CompressedParametersRecord( plan );
            }
        } );
        map.put( 12, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new SparsenessParametersRecord( plan );
            }
        } );
        map.put( 13, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new CompressedVariableValuesRecord( plan );
            }
        } );
        map.put( -1, new TypedRecordFactory() {
            public Record createRecord( RecordPlan plan ) throws IOException {
                return new UnusedInternalRecord( plan );
            }
        } );
        int[] recTypes = new int[ map.size() ];
        int irt = 0;
        for ( int recType : map.keySet() ) {
            recTypes[ irt++ ] = recType;
        }
        Arrays.sort( recTypes );
        assert Arrays.equals( recTypes, new int[] { -1, 1, 2, 3, 4, 5, 6, 7,
                                                    8, 9, 10, 11, 12, 13 } );
        return Collections.unmodifiableMap( map );
    }

    /**
     * Object which can generate a particular record type from a plan.
     */
    private static interface TypedRecordFactory<R extends Record> {

        /**
         * Creates a record from bytes.
         *
         * @param   plan  basic record information
         * @return   record
         */
        R createRecord( RecordPlan plan ) throws IOException;
    }
}
