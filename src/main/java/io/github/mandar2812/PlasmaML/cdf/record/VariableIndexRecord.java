package io.github.mandar2812.PlasmaML.cdf.record;

import io.github.mandar2812.PlasmaML.cdf.Buf;
import io.github.mandar2812.PlasmaML.cdf.CdfField;
import io.github.mandar2812.PlasmaML.cdf.OffsetField;

import java.io.IOException;

/**
 * Field data for CDF record of type Variable Index Record.
 *
 * @author   Mark Taylor
 * @since    19 Jun 2013
 */
public class VariableIndexRecord extends Record {

    @CdfField
    @OffsetField
    public final long vxrNext;
    @CdfField public final int nEntries;
    @CdfField public final int nUsedEntries;
    @CdfField public final int[] first;
    @CdfField public final int[] last;
    @CdfField @OffsetField public final long[] offset;

    /** 
     * Constructor.
     *
     * @param  plan   basic record information
     */
    public VariableIndexRecord( RecordPlan plan ) throws IOException {
        super( plan, "VXR", 6 );
        Buf buf = plan.getBuf();
        Pointer ptr = plan.createContentPointer();
        this.vxrNext = buf.readOffset( ptr );
        this.nEntries = buf.readInt( ptr );
        this.nUsedEntries = buf.readInt( ptr );
        this.first = readIntArray( buf, ptr, this.nEntries );
        this.last = readIntArray( buf, ptr, this.nEntries );
        this.offset = readOffsetArray( buf, ptr, this.nEntries );
        checkEndRecord( ptr );
    }
}
