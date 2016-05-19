package io.github.mandar2812.PlasmaML.cdf.record;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import io.github.mandar2812.PlasmaML.cdf.Buf;
import io.github.mandar2812.PlasmaML.cdf.CdfFormatException;

/**
 * Keeps track of where a variable's record data is stored.
 *
 * <p>To work out the buffer and offset from which to read a record value,
 * you can do something like this:
 * <pre>
 *     int ient = recMap.getEntryIndex(irec);
 *     Object value =
 *            ient &gt;= 0
 *          ? readBuffer(recMap.getBuf(ient), recMap.getOffset(ient,irec))
 *          : NO_STORED_VALUE;
 * </pre>
 *       
 *
 * @author   Mark Taylor
 * @since    21 Jun 2013
 */
public class RecordMap {

    private final int nent_;
    private final int[] firsts_;
    private final int[] lasts_;
    private final Buf[] bufs_;
    private final long[] offsets_;
    private final int recSize_;
    private Block lastBlock_;

    /**
     * Constructor.
     *
     * @param   array of entries containing stored variable record blocks,
     *          need not be sorted
     * @param   recSize   size of each variable record in bytes
     */
    private RecordMap( Entry[] entries, int recSize ) {
        recSize_ = recSize;

        // Sort entries into order of record data.
        Arrays.sort( entries );

        // Store the entry information in a convenient form.
        nent_ = entries.length;
        firsts_ = new int[ nent_ ];
        lasts_ = new int[ nent_ ];
        bufs_ = new Buf[ nent_ ];
        offsets_ = new long[ nent_ ];
        for ( int ie = 0; ie < nent_; ie++ ) {
            Entry entry = entries[ ie ];
            firsts_[ ie ] = entry.first_;
            lasts_[ ie ] = entry.last_;
            bufs_[ ie ] = entry.buf_;
            offsets_[ ie ] = entry.offset_;
        }

        // Initialise the most recently used block value
        lastBlock_ = nent_ > 0 ? calculateBlock( 0 )
                               : new Block( -1, -1, -1 );
    }

    /**
     * Returns the number of entries managed by this map.
     *
     * @return   entry count
     */
    public int getEntryCount() {
        return nent_;
    }

    /**
     * Returns the index of the entry containing a given record.
     * If one of the entries contains the given record, return its index.
     * If no entry contains it (the record is in a sparse region),
     * return <code>(-fr-2)</code>, where <code>fr</code>
     * is the index of the previous entry.
     * A value of -1 indicates that the requested record is
     * in a sparse region before the first stored record.
     *
     * <p>If non-negative, the result can be used with the
     * <code>getBuf</code> and <code>getOffset</code> methods.
     *
     * @param  irec  record index
     * @return  index of entry covering <code>irec</code>, or a negative
     *          value if no entry covers it
     */
    public synchronized int getEntryIndex( int irec ) {

        // There's a good chance that the answer is the same as the last
        // time somebody asked, so first of all do the cheap test to find
        // out if that's the case.  If so, return the cached one.
        // Otherwise, do the work to find out the right answer.
        if ( ! lastBlock_.contains( irec ) ) {
            lastBlock_ = calculateBlock( irec );
        }
        assert lastBlock_.contains( irec );
        return lastBlock_.ient_;
    }

    /**
     * Returns the data buffer for a given entry.
     * The entry index must correspond to an actual entry,
     * that is it must not be negative.
     *
     * @param  ient  entry index
     * @return  buf
     * @see   #getEntryIndex
     */
    public Buf getBuf( int ient ) {
        return bufs_[ ient ];
    }

    /**
     * Returns the byte offset for a record in a given entry.
     * The <code>ient</code> parameter must reference an actual entry
     * (it must be non-negative), and that entry must contain
     * the given record <code>irec</code>,
     *
     * @param   ient  entry index for entry containing <code>irec</code>
     * @param   irec  record index
     * @return   offset into the entry's buffer at which <code>irec</code>
     *           can be found
     * @see   #getEntryIndex
     */
    public long getOffset( int ient, int irec ) {
        assert irec >= firsts_[ ient ] && irec <= lasts_[ ient ];
        return offsets_[ ient ] + ( irec - firsts_[ ient ] ) * recSize_;
    }

    /**
     * Returns the offset of the last record in a given entry.
     *
     * @param  ient  non-negative entry index
     * @return  offset into ient's buffer of ient's final record
     */
    public long getFinalOffsetInEntry( int ient ) {
        return offsets_[ ient ]
             + ( lasts_[ ient ] - firsts_[ ient ] + 1 ) * recSize_;
    }

    /**
     * Examines this map's lookup tables to determine the block covering
     * a given record.
     *
     * @param  irec   record index
     * @return   block containing irec
     */
    private Block calculateBlock( int irec ) {

        // Look for the record in the first-record-of-entry list.
        int firstIndex = binarySearch( firsts_, irec );

        // If found, irec is in the corresponding block.
        if ( firstIndex >= 0 ) {
            return new Block( firstIndex,
                              firsts_[ firstIndex ], lasts_[ firstIndex ] );
        }

        // If it's located before the start, it's in a sparse block
        // before the first actual record.
        else if ( firstIndex == -1 ) {
            return new Block( -firstIndex - 2, 0, firsts_[ 0 ] - 1 );
        }

        // Otherwise, record the first entry it's after the start of.
        else {
            firstIndex = -2 - firstIndex;
        }

        // Look for the record in the last-record-of-entry list.
        int lastIndex = binarySearch( lasts_, irec );

        // If found, irec is in the corresponding block.
        if ( lastIndex >= 0 ) {
            return new Block( lastIndex,
                              firsts_[ lastIndex ], lasts_[ lastIndex ] );
        }

        // If it's located after the end, it's in a sparse block
        // after the last actual record.
        else if ( lastIndex == - nent_ - 1 ) {
            return new Block( lastIndex,
                              lasts_[ nent_ - 1 ], Integer.MAX_VALUE );
        }

        // Otherwise, record the last entry it's before the end of.
        else {
            lastIndex = -1 - lastIndex;
        }

        // If it's after the first record and before the last record
        // of a single block, that's the one.
        if ( firstIndex == lastIndex ) {
            return new Block( firstIndex,
                              firsts_[ firstIndex ], lasts_[ firstIndex ] );
        }

        // Otherwise, it's in a sparse block between
        // the end of the entry it's after the first record of, and
        // the start of the entry it's before the last record of.
        else {
            return new Block( -firstIndex - 2,
                              lasts_[ firstIndex ] + 1,
                              firsts_[ lastIndex ] - 1 );
        }
    }

    /**
     * Returns a record map for a given variable.
     *
     * @param  vdr  variable descriptor record
     * @param  recFact  record factory
     * @param  recSize  size in bytes of each variable value record
     * @return  record map
     */
    public static RecordMap createRecordMap( VariableDescriptorRecord vdr,
                                             RecordFactory recFact,
                                             int recSize )
            throws IOException {
        Compression compress = getCompression( vdr, recFact );
        Buf buf = vdr.getBuf();

        // Walk the entry linked list to assemble a list of entries.
        List<Entry> entryList = new ArrayList<Entry>();
        for ( long vxrOffset = vdr.vxrHead; vxrOffset != 0; ) {
            VariableIndexRecord vxr =
                recFact.createRecord( buf, vxrOffset,
                                      VariableIndexRecord.class );
            readEntries( vxr, buf, recFact, recSize, compress, entryList );
            vxrOffset = vxr.vxrNext;
        }
        Entry[] entries = entryList.toArray( new Entry[ 0 ] );

        // Make a RecordMap out of it.
        return new RecordMap( entries, recSize );
    }

    /**
     * Returns the compression type for a given variable.
     *
     * @param  vdr  variable descriptor record
     * @param  recFact  record factory
     * @return  compression type, not null but may be NONE
     */
    private static Compression getCompression( VariableDescriptorRecord vdr,
                                               RecordFactory recFact )
            throws IOException {
        boolean hasCompress = Record.hasBit( vdr.flags, 2 );
        if ( hasCompress && vdr.cprOrSprOffset != -1 ) {
            CompressedParametersRecord cpr =
                recFact.createRecord( vdr.getBuf(), vdr.cprOrSprOffset,
                                      CompressedParametersRecord.class );
            return Compression.getCompression( cpr.cType );
        }
        else {
            return Compression.NONE;
        }
    }

    /**
     * Reads the list of Entries from a Variable Index Record
     * into a supplied list.
     *
     * @param  vxr  variable index record
     * @param  buf  data buffer containing vxr
     * @param  recFact  record factory
     * @param  recSize  size in bytes of each variable value record
     * @param  compress  compression type
     * @param   list  list into which any entries found are added
     */
    private static void readEntries( VariableIndexRecord vxr, Buf buf,
                                     RecordFactory recFact, int recSize,
                                     Compression compress, List<Entry> list )
            throws IOException {

        // Go through each entry in the VXR.
        // Each one may be a VVR, a CVVR, or a subordinate VXR
        // (the format document is not very explicit about this, but it
        // seems to be what happens).
        // The only way to know which each entry is, is to examine
        // the record type value for each one (the RecordFactory takes
        // care of this by creating the right class).
        int nent = vxr.nUsedEntries;
        for ( int ie = 0; ie < nent; ie++ ) {
            int first = vxr.first[ ie ];
            int last = vxr.last[ ie ];
            Record rec = recFact.createRecord( buf, vxr.offset[ ie ] );

            // VVR: turn it directly into a new Entry and add to the list.
            if ( rec instanceof VariableValuesRecord ) {
                VariableValuesRecord vvr = (VariableValuesRecord) rec;
                list.add( new Entry( first, last, buf,
                                     vvr.getRecordsOffset() ) );
            }

            // CVVR: uncompress and turn it into a new Entry and add to list.
            else if ( rec instanceof CompressedVariableValuesRecord ) {
                CompressedVariableValuesRecord cvvr =
                    (CompressedVariableValuesRecord) rec;
                int uncompressedSize = ( last - first + 1 ) * recSize;
                Buf cBuf = Bufs.uncompress( compress, buf, cvvr.getDataOffset(),
                                            uncompressedSize );
                list.add( new Entry( first, last, cBuf, 0L ) );
            }

            // VXR: this is a reference to another sub-tree of entries.
            // Handle it with a recursive call to this routine.
            else if ( rec instanceof VariableIndexRecord ) {

                // Amazingly, it's necessary to walk both the subtree of
                // VXRs hanging off the entry list *and* the linked list
                // of VXRs whose head is contained in this record.
                // This does seem unnecessarily complicated, but I've
                // seen at least one file where it happens
                // (STEREO_STA_L1_MAG_20070708_V03.cdf).
                VariableIndexRecord subVxr = (VariableIndexRecord) rec;
                readEntries( subVxr, buf, recFact, recSize, compress, list );
                for ( long nextVxrOff = subVxr.vxrNext; nextVxrOff != 0; ) {
                    VariableIndexRecord nextVxr =
                        recFact.createRecord( buf, nextVxrOff,
                                              VariableIndexRecord.class );
                    readEntries( nextVxr, buf, recFact, recSize, compress,
                                 list );
                    nextVxrOff = nextVxr.vxrNext;
                }
            }

            // Some other record type - no ideas.
            else {
                String msg = new StringBuffer()
                   .append( "Unexpected record type (" )
                   .append( rec.getRecordType() )
                   .append( ") pointed to by VXR offset" )
                   .toString();
                throw new CdfFormatException( msg );
            }
        }
    }

    /**
     * Represents an entry in a Variable Index Record.
     * It records the position and extent of a contiguous block of
     * variable values (a Variable Values Record) for its variable.
     *
     * <p>Note that following the usage in VXR fields, the first and
     * last values are inclusive, so the number of records represented
     * by this entry is <code>last-first+1</code>.
     */
    private static class Entry implements Comparable<Entry> {
        private final int first_;
        private final int last_;
        private final Buf buf_;
        private final long offset_;

        /**
         * Constructor.
         *
         * @param  first  index of first record in this entry
         * @param  last   index of last record (inclusive) in this entry
         * @param  buf    buffer containing the data
         * @param  offset  byte offset into buffer at which the record block
         *                 starts
         */
        Entry( int first, int last, Buf buf, long offset ) {
            first_ = first;
            last_ = last;
            buf_ = buf;
            offset_ = offset;
        }

        /**
         * Compares this entry to another on the basis of record indices.
         */
        public int compareTo( Entry other ) {
            return this.first_ - other.first_;
        }
    }

    /**
     * Represents a block of records, that is a contiguous sequence of records.
     * This may corrrespond to an actual data-bearing Entry, or it may
     * correspond to a gap where no Entry exists, before the first entry,
     * or after the last, or between entries if the records are sparse.
     *
     * <p>The <code>ient</code> member gives the index of the corresponding
     * Entry.
     * If there is no corresponding entry (the record is in a sparse
     * region), the value is <code>(-fr-2)</code>, where <code>fr</code>
     * is the index of the previous entry.
     * A value of -1 indicates that the requested record is
     * in a sparse region before the first stored record.
     *
     * <p>Note that following the usage in VXR fields, the low and
     * high values are inclusive, so the number of records represented
     * by this entry is <code>high-low+1</code>.
     *
     */
    private static class Block {
        final int ient_;
        final int low_;
        final int high_;

        /**
         * Constructor.
         *
         * @param   ient  index of Entry containing this block's data;
         *                negative value means sparse
         * @param   low   lowest record index contained in this block
         * @param   high  highest record index contained in this block
         */
        Block( int ient, int low, int high ) {
            ient_ = ient;
            low_ = low;
            high_ = high;
        }
     
        /**
         * Indicates whether a given record falls within this block.
         *
         * @param  irec  record index
         * @return  true iff irec is covered by this block
         */
        boolean contains( int irec ) {
            return irec >= low_ && irec <= high_;
        }
    }

    /**
     * Performs a binary search on an array.
     * Calls Arrays.binarySearch to do the work.
     *
     * @param  array   array in ascending sorted order
     * @param  key   value to search for
     * @return  index of the search key, if it is contained in the list;
     *          otherwise, (-(insertion point) - 1). 
     * @see   java.util.Arrays#binarySearch(int[],int)
     */
    private static int binarySearch( int[] array, int key ) {
        assert isSorted( array );
        return Arrays.binarySearch( array, key );
    }

    /**
     * Determines whether an integer array is currently sorted in
     * ascending (well, non-descending) order.
     *
     * @param  values  array
     * @return  true iff sorted
     */
    private static boolean isSorted( int[] values ) {
        int nval = values.length;
        for ( int i = 1; i < nval; i++ ) {
            if ( values[ i ] < values[ i - 1 ] ) {
                return false;
            }
        }
        return true;
    }
}
