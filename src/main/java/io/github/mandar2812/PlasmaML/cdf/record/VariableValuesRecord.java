package io.github.mandar2812.PlasmaML.cdf.record;

/**
 * Field data for CDF record of type Variable Values Record.
 *
 * @author   Mark Taylor
 * @since    19 Jun 2013
 */
public class VariableValuesRecord extends Record {

    private final long recordsOffset_;

    /**
     * Constructor.
     *
     * @param   plan  basic record information
     */
    public VariableValuesRecord( RecordPlan plan ) {
        super( plan, "VVR", 7 );
        Pointer ptr = plan.createContentPointer();
        recordsOffset_ = ptr.get();
    }

    /**
     * Returns the file offset at which the records data in this record
     * starts.
     *
     * @return  file offset for start of Records field
     */
    public long getRecordsOffset() {
        return recordsOffset_;
    }
}
