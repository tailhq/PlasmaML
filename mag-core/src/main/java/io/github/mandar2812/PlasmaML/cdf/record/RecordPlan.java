package io.github.mandar2812.PlasmaML.cdf.record;

import io.github.mandar2812.PlasmaML.cdf.Buf;

/**
 * Records basic information about the position, extent and type of
 * a CDF record.
 *
 * @author   Mark Taylor
 * @since    18 Jun 2013
 */
public class RecordPlan {

    private final long start_;
    private final long recSize_;
    private final int recType_;
    private final Buf buf_;
    
    /**
     * Constructor.
     *
     * @param   start   offset into buffer of record start
     * @param   recSize  number of bytes comprising record
     * @param   recType  integer record type field
     * @param   buf     buffer containing record bytes
     */
    public RecordPlan( long start, long recSize, int recType, Buf buf ) {
        start_ = start;
        recSize_ = recSize;
        recType_ = recType;
        buf_ = buf;
    }

    /**
     * Returns the size of the record in bytes.
     *
     * @return  record size
     */
    public long getRecordSize() {
        return recSize_;
    }

    /**
     * Returns the type code identifying what kind of CDF record it is.
     *
     * @return   record type
     */
    public int getRecordType() {
        return recType_;
    }

    /**
     * Returns the buffer containing the record data.
     *
     * @return  buffer
     */
    public Buf getBuf() {
        return buf_;
    }

    /**
     * Returns a pointer initially pointing at the first content byte of
     * the record represented by this plan.
     * This is the first item after the RecordSize and RecordType items
     * that always appear first in a CDF record, and whose values are
     * known by this object.
     *
     * @return  pointer pointing at the start of the record-type-specific
     *          content
     */
    public Pointer createContentPointer() {
        long pos = start_;
        pos += buf_.isBit64() ? 8 : 4;  // record size
        pos += 4;                       // record type
        return new Pointer( pos );
    }

    /**
     * Returns the number of bytes in this record read (or skipped) by the
     * current state of a given pointer.
     *
     * @param   ptr  pointer
     * @return  number of bytes between record start and pointer value
     */
    public long getReadCount( Pointer ptr ) {
        return ptr.get() - start_;
    }
}
