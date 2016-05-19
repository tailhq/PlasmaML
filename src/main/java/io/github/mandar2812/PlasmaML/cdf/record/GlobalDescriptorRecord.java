package io.github.mandar2812.PlasmaML.cdf.record;

import io.github.mandar2812.PlasmaML.cdf.Buf;
import io.github.mandar2812.PlasmaML.cdf.CdfField;
import io.github.mandar2812.PlasmaML.cdf.OffsetField;

import java.io.IOException;

/**
 * Field data for CDF record of type Global Descriptor Record.
 *
 * @author   Mark Taylor
 * @since    19 Jun 2013
 */
public class GlobalDescriptorRecord extends Record {

    @CdfField
    @OffsetField
    public final long rVdrHead;
    @CdfField @OffsetField public final long zVdrHead;
    @CdfField @OffsetField public final long adrHead;
    @CdfField public final long eof;
    @CdfField public final int nrVars;
    @CdfField public final int numAttr;
    @CdfField public final int rMaxRec;
    @CdfField public final int rNumDims;
    @CdfField public final int nzVars;
    @CdfField @OffsetField public final long uirHead;
    @CdfField public final int rfuC;
    @CdfField public final int leapSecondLastUpdated;
    @CdfField public final int rfuE;
    @CdfField public final int[] rDimSizes;

    /**
     * Constructor.
     *
     * @param  plan   basic record information
     */
    public GlobalDescriptorRecord( RecordPlan plan ) throws IOException {
        super( plan, "GDR", 2 );
        Buf buf = plan.getBuf();
        Pointer ptr = plan.createContentPointer();
        this.rVdrHead = buf.readOffset( ptr );
        this.zVdrHead = buf.readOffset( ptr );
        this.adrHead = buf.readOffset( ptr );
        this.eof = buf.readOffset( ptr );
        this.nrVars = buf.readInt( ptr );
        this.numAttr = buf.readInt( ptr );
        this.rMaxRec = buf.readInt( ptr );
        this.rNumDims = buf.readInt( ptr );
        this.nzVars = buf.readInt( ptr );
        this.uirHead = buf.readOffset( ptr );
        this.rfuC = checkIntValue( buf.readInt( ptr ), 0 );
        this.leapSecondLastUpdated = buf.readInt( ptr );
        this.rfuE = checkIntValue( buf.readInt( ptr ), -1 );
        this.rDimSizes = readIntArray( buf , ptr, this.rNumDims );
        checkEndRecord( ptr );
    }
}
