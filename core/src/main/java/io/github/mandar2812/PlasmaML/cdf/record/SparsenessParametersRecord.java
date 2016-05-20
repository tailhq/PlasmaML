package io.github.mandar2812.PlasmaML.cdf.record;

import io.github.mandar2812.PlasmaML.cdf.Buf;
import io.github.mandar2812.PlasmaML.cdf.CdfField;

import java.io.IOException;

/**
 * Field data for CDF record of type Sparseness Parameters Record.
 *
 * @author   Mark Taylor
 * @since    19 Jun 2013
 */
public class SparsenessParametersRecord extends Record {

    @CdfField
    public final int sArraysType;
    @CdfField public final int rfuA;
    @CdfField public final int pCount;
    @CdfField public final int[] sArraysParms;

    /**
     * Constructor.
     *
     * @param   plan  basic record information
     */
    public SparsenessParametersRecord( RecordPlan plan ) throws IOException {
        super( plan, "SPR", 12 );
        Buf buf = plan.getBuf();
        Pointer ptr = plan.createContentPointer();
        this.sArraysType = buf.readInt( ptr );
        this.rfuA = checkIntValue( buf.readInt( ptr ), 0 );
        this.pCount = buf.readInt( ptr );
        this.sArraysParms = readIntArray( buf, ptr, this.pCount );
        checkEndRecord( ptr );
    }
}
