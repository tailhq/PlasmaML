package io.github.mandar2812.PlasmaML.cdf.record;

import io.github.mandar2812.PlasmaML.cdf.Buf;
import io.github.mandar2812.PlasmaML.cdf.CdfField;
import io.github.mandar2812.PlasmaML.cdf.OffsetField;

import java.io.IOException;

/**
 * Field data for CDF record of type Unused Internal Record.
 *
 * @author   Mark Taylor
 * @since    19 Jun 2013
 */
public class UnusedInternalRecord extends Record {

    @CdfField
    @OffsetField
    public final long nextUir;
    @CdfField @OffsetField public final long prevUir;

    /**
     * Constructor.
     *
     * @param  plan   basic record information
     */
    public UnusedInternalRecord( RecordPlan plan ) throws IOException {
        super( plan, "UIR", -1 );
        Buf buf = plan.getBuf();
        Pointer ptr = plan.createContentPointer();
        int planHeaderSize = (int) plan.getReadCount( ptr );

        // This UIR may be unsociable and too small to contain UIR fields.
        // If so, don't attempt to read them (if we are extremely unlucky
        // they might be off the end of the file).  Check the record size
        // is large enough to accommodate these fields before reading them.
        int pointerSize = buf.isBit64() ? 8 : 4;
        int sociableUirSize = planHeaderSize + 2 * pointerSize;
        if ( plan.getRecordSize() >= sociableUirSize ) {
            this.nextUir = buf.readOffset( ptr );
            this.prevUir = buf.readOffset( ptr );
        }
        else {  // too small to be sociable
            this.nextUir = -1L;
            this.prevUir = -1L;
        }
    }
}
