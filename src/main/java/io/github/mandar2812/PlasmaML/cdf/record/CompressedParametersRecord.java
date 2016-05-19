package io.github.mandar2812.PlasmaML.cdf.record;

import io.github.mandar2812.PlasmaML.cdf.Buf;
import io.github.mandar2812.PlasmaML.cdf.CdfField;

import java.io.IOException;

/**
 * Field data for CDF record of type Compressed Parameters Record.
 *
 * @author   Mark Taylor
 * @since    19 Jun 2013
 */
public class CompressedParametersRecord extends Record {

    @CdfField
    public final int cType;
    @CdfField public final int rfuA;
    @CdfField public final int pCount;
    @CdfField public final int[] cParms;

    /**
     * Constructor.
     *
     * @param  plan   basic record information
     */
    public CompressedParametersRecord( RecordPlan plan ) throws IOException {
        super( plan, "CPR", 11 );
        Buf buf = plan.getBuf();
        Pointer ptr = plan.createContentPointer();
        this.cType = buf.readInt( ptr );
        this.rfuA = checkIntValue( buf.readInt( ptr ), 0 );
        this.pCount = buf.readInt( ptr );
        this.cParms = readIntArray( buf, ptr, this.pCount );
        checkEndRecord( ptr );
    }
}
