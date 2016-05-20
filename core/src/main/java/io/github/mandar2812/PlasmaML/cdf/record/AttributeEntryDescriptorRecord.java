package io.github.mandar2812.PlasmaML.cdf.record;

import io.github.mandar2812.PlasmaML.cdf.Buf;
import io.github.mandar2812.PlasmaML.cdf.CdfField;
import io.github.mandar2812.PlasmaML.cdf.OffsetField;

import java.io.IOException;

/**
 * Abstract superclass for CDF Attribute Entry Descriptor Records.
 * Two concrete subclasses exist for AzEDRs and AgrEDRs.
 *
 * @author   Mark Taylor
 * @since    19 Jun 2013
 */
public abstract class AttributeEntryDescriptorRecord extends Record {

    @CdfField
    @OffsetField
    public final long aedrNext;
    @CdfField public final int attrNum;
    @CdfField public final int dataType;
    @CdfField public final int num;
    @CdfField public final int numElems;
    @CdfField public final int rfuA;
    @CdfField public final int rfuB;
    @CdfField public final int rfuC;
    @CdfField public final int rfuD;
    @CdfField public final int rfuE;
    private final long valueOffset_;

    /**
     * Constructor.
     *
     * @param  plan  basic record info
     * @param  abbrev  abbreviated name for record type
     * @param  recordType  record type code
     */
    private AttributeEntryDescriptorRecord( RecordPlan plan, String abbrev,
                                            int recordType )
            throws IOException {
        super( plan, abbrev, recordType );
        Buf buf = plan.getBuf();
        Pointer ptr = plan.createContentPointer();
        this.aedrNext = buf.readOffset( ptr );
        this.attrNum = buf.readInt( ptr );
        this.dataType = buf.readInt( ptr );
        this.num = buf.readInt( ptr );
        this.numElems = buf.readInt( ptr );
        this.rfuA = checkIntValue( buf.readInt( ptr ), 0 );
        this.rfuB = checkIntValue( buf.readInt( ptr ), 0 );
        this.rfuC = checkIntValue( buf.readInt( ptr ), 0 );
        this.rfuD = checkIntValue( buf.readInt( ptr ), -1 );
        this.rfuE = checkIntValue( buf.readInt( ptr ), -1 );
        valueOffset_ = ptr.get();
    }

    /**
     * Returns the file offset at which this record's Value field starts.
     *
     * @return  file offset of Value field
     */
    public long getValueOffset() {
        return valueOffset_;
    }

    /**
     * Field data for CDF record of type Attribute g/rEntry Descriptor Record.
     */
    public static class GrVariant extends AttributeEntryDescriptorRecord {

        /**
         * Constructor.
         *
         * @param  plan   basic record information
         */
        public GrVariant( RecordPlan plan ) throws IOException {
            super( plan, "AgrEDR", 5 );
        }
    }

    /**
     * Field data for CDF record of type Attribute zEntry Descriptor Record.
     */
    public static class ZVariant extends AttributeEntryDescriptorRecord {

        /**
         * Constructor.
         *
         * @param  plan   basic record information
         */
        public ZVariant( RecordPlan plan ) throws IOException {
            super( plan, "AzEDR", 9 );
        }
    }
}
