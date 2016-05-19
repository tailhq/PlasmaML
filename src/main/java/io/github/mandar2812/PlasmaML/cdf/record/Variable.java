package io.github.mandar2812.PlasmaML.cdf.record;

import java.io.IOException;
import java.lang.reflect.Array;

import io.github.mandar2812.PlasmaML.cdf.*;
import io.github.mandar2812.PlasmaML.cdf.Buf;

/**
 * Provides the metadata and record data for a CDF variable.
 *
 * <p>At construction time, a map of where the records are stored is
 * constructed, but the record data itself is not read unless or until
 * one of the <code>read</code> methods is called.
 *
 * <p>This interface does not currently support data reading in such
 * a flexible way as the official CDF interface.
 * You can read a record's worth of data at a time using either
 * {@link #readRawRecord readRawRecord} (which should be fairly efficient) or
 * {@link #readShapedRecord readShapedRecord} (which may have to copy and
 * possibly re-order the array, and may not be so efficient).
 *
 * @author   Mark Taylor
 * @since    20 Jun 2013
 */
public class Variable {

    private final VariableDescriptorRecord vdr_;
    private final Buf buf_;
    private final RecordFactory recFact_;
    private final boolean isZVariable_;
    private final boolean recordVariance_;
    private final Shaper shaper_;
    private final int rvaleng_;
    private final DataType dataType_;
    private final DataReader dataReader_;
    private final Object padRawValueArray_;
    private final Object shapedPadValueRowMajor_;
    private final Object shapedPadValueColumnMajor_;
    private final String summaryTxt_;
    private RecordReader recordReader_;

    /**
     * Constructor.
     *
     * @param   vdr   variable descriptor record for the variable
     * @param   cdfInfo  global CDF information
     * @param   recFact  record factory
     */
    public Variable( VariableDescriptorRecord vdr, CdfInfo cdfInfo,
                     RecordFactory recFact ) throws IOException {

        // Prepare state for reading data.
        vdr_ = vdr;
        buf_ = vdr.getBuf();
        recFact_ = recFact;
        isZVariable_ = vdr.getRecordType() == 8;
        dataType_ = DataType.getDataType( vdr.dataType, cdfInfo );
        recordVariance_ = Record.hasBit( vdr_.flags, 0 );
        int[] dimSizes = isZVariable_ ? vdr.zDimSizes : cdfInfo.getRDimSizes();
        boolean[] dimVarys = vdr.dimVarys;
        boolean rowMajor = cdfInfo.getRowMajor();
        int numElems = vdr.numElems;

        // As far as I understand the internal formats document, only
        // character data types can have numElems>1 here.
        assert dataType_.hasMultipleElementsPerItem() || numElems == 1;
        shaper_ =
            Shaper.createShaper( dataType_, dimSizes, dimVarys, rowMajor );
        int nraw = shaper_.getRawItemCount();
        dataReader_ = new DataReader( dataType_, numElems, nraw );
        rvaleng_ = Array.getLength( dataReader_.createValueArray() );

        // Read pad value if present.
        long padOffset = vdr.getPadValueOffset();
        if ( padOffset >= 0 ) {
            DataReader padReader = new DataReader( dataType_, numElems, 1 );
            assert vdr.getPadValueSize() == padReader.getRecordSize();
            Object padValueArray = padReader.createValueArray();
            padReader.readValue( buf_, padOffset, padValueArray );
            Object rva = dataReader_.createValueArray();
            int ngrp = dataType_.getGroupSize();
            for ( int i = 0; i < nraw; i++ ) {
                System.arraycopy( padValueArray, 0, rva, i * ngrp, ngrp );
            }
            padRawValueArray_ = rva;
            shapedPadValueRowMajor_ = shaper_.shape( padRawValueArray_, true );
            shapedPadValueColumnMajor_ =
                shaper_.shape( padRawValueArray_, false );
        }
        else if ( vdr_.sRecords != 0 ) {
            Object padValueArray = dataType_.getDefaultPadValueArray();
            Object rva = dataReader_.createValueArray();
            int ngrp = dataType_.getGroupSize();
            for ( int i = 0; i < nraw; i++ ) {
                System.arraycopy( padValueArray, 0, rva, i * ngrp, ngrp );
            }
            padRawValueArray_ = rva;
            shapedPadValueRowMajor_ = shaper_.shape( padRawValueArray_, true );
            shapedPadValueColumnMajor_ = shapedPadValueRowMajor_;
        }
        else {
            padRawValueArray_ = null;
            shapedPadValueRowMajor_ = null;
            shapedPadValueColumnMajor_ = null;
        }

        // Assemble a short summary string.
        String shapeTxt = "";
        String varyTxt = "";
        for ( int idim = 0; idim < dimSizes.length; idim++ ) {
            if ( idim > 0 ) {
                shapeTxt += ',';
            }
            shapeTxt += dimSizes[ idim ];
            varyTxt += dimVarys[ idim ] ? 'T' : 'F';
        }
        summaryTxt_ = new StringBuffer()
            .append( dataType_.getName() )
            .append( ' ' )
            .append( isZVariable_ ? "(z)" : "(r)" )
            .append( ' ' )
            .append( dimSizes.length )
            .append( ':' )
            .append( '[' )
            .append( shapeTxt )
            .append( ']' )
            .append( ' ' )
            .append( recordVariance_ ? 'T' : 'F' )
            .append( '/' )
            .append( varyTxt )
            .toString();
    }

    /**
     * Returns this variable's name.
     *
     * @return   variable name
     */
    public String getName() {
        return vdr_.name;
    }

    /**
     * Returns the index number within the CDF of this variable.
     *
     * @return  variable num
     */
    public int getNum() {
        return vdr_.num;
    }

    /**
     * Indicates whether this variable is a zVariable or rVariable.
     *
     * @return   true for zVariable, false for rVariable
     */
    public boolean isZVariable() {
        return isZVariable_;
    }

    /**
     * Returns the upper limit of records that may have values.
     * The actual number of records may be lower than this in case of sparsity.
     *
     * @return   maximum record count
     */
    public int getRecordCount() {
        return vdr_.maxRec + 1;
    }

    /**
     * Returns the data type of this variable.
     *
     * @return  data type
     */
    public DataType getDataType() {
        return dataType_;
    }

    /**
     * Returns an object that knows about the array dimensions
     * of the data values.
     *
     * @return  shaper
     */
    public Shaper getShaper() {
        return shaper_;
    }

    /**
     * Indicates whether this variable has a value which is fixed for all
     * records or can vary per record.
     *
     * @return   false for fixed, true for varying
     */
    public boolean getRecordVariance() {
        return recordVariance_;
    }

    /**
     * Returns a short text string describing the type, shape and variance
     * of this variable.
     *
     * @return  text summary of variable characteristics
     */
    public String getSummary() {
        return summaryTxt_;
    }

    /**
     * Returns the VariableDescriptorRecord on which this Variable instance
     * is based.
     *
     * @return  variable descriptor record (rVDR or zVDR)
     */
    public VariableDescriptorRecord getDescriptor() {
        return vdr_;
    }

    /**
     * Creates a workspace array suitable for use with this variable's
     * reading methods.
     * The returned array is a 1-dimensional array of a primitive type
     * or of String.
     *
     * @return  workspace array for data reading
     */
    public Object createRawValueArray() {
        return dataReader_.createValueArray();
    }

    /**
     * Indicates whether a real distinct file-based record exists for
     * the given index.
     * Reading a record will give you a result in any case, but if this
     * returns false it will be some kind of fixed or default value.
     *
     * @param   irec  record index
     * @return   true iff a file-based record exists for irec
     */
    public boolean hasRecord( int irec ) throws IOException {
        return getRecordReader().hasRecord( irec );
    }

    /**
     * Reads the data from a single record into a supplied raw value array.
     * The values are read into the supplied array in the order in which
     * they are stored in the data stream, that is depending on the row/column
     * majority of the CDF.
     * <p>The raw value array is as obtained from {@link #createRawValueArray}.
     *
     * @param  irec  record index
     * @param  rawValueArray  workspace array, as created by the
     *                        <code>createRawValueArray</code> method
     */
    public void readRawRecord( int irec, Object rawValueArray )
             throws IOException {
         getRecordReader().readRawRecord( irec, rawValueArray );
    }

    /**
     * Reads the data from a single record and returns it as an object
     * of a suitable type for this variable.
     * If the variable type a scalar, then the return value will be
     * one of the primitive wrapper types (Integer etc),
     * otherwise it will be an array of primitive or String values.
     * If the majority of the stored data does not match the
     * <code>rowMajor</code> argument, the array elements will be
     * rordered appropriately.
     * If some of the dimension variances are false, the values will
     * be duplicated accordingly.
     * The Shaper returned from the {@link #getShaper} method
     * can provide more information on the return value from this method.
     *
     * <p>The workspace is as obtained from {@link #createRawValueArray}.
     *
     * @param  irec  record index
     * @param  rowMajor  required majority of output array; true for row major,
     *                   false for column major; only has an effect for
     *                   dimensionality &gt;=2
     * @param  rawValueArrayWorkspace  workspace array, as created by the
     *                                 <code>createRawValueArray</code> method
     * @return   a new object containing the shaped result
     *           (not the same object as <code>rawValueArray</code>
     */
    public Object readShapedRecord( int irec, boolean rowMajor,
                                    Object rawValueArrayWorkspace )
             throws IOException {
         return getRecordReader()
               .readShapedRecord( irec, rowMajor, rawValueArrayWorkspace );
    }

    /**
     * Returns an object that can read records for this variable.
     * Constructing it requires reading maps of where the record values
     * are stored, which might in principle involve a bit of work,
     * so do it lazily.
     *
     * @return  record reader
     */
    private synchronized RecordReader getRecordReader() throws IOException {
        if ( recordReader_ == null ) {
            recordReader_ = createRecordReader();
        }
        return recordReader_;
    }

    /**
     * Constructs a record reader.
     *
     * @return  new record reader
     */
    private RecordReader createRecordReader() throws IOException {
        RecordMap recMap =
            RecordMap.createRecordMap( vdr_, recFact_,
                                       dataReader_.getRecordSize() );
        if ( ! recordVariance_ ) {
            return new NoVaryRecordReader( recMap );
        }
        else {
            // Get sparse records type.  This is missing from the CDF Internal
            // Format Description document, but cdf.h says:
            //    #define NO_SPARSERECORDS                0L
            //    #define PAD_SPARSERECORDS               1L
            //    #define PREV_SPARSERECORDS              2L
            int sRecords = vdr_.sRecords;
            if ( sRecords == 0 ) {
                return new UnsparseRecordReader( recMap );
            }
            else if ( sRecords == 1 ) {
                assert padRawValueArray_ != null;
                return new PadRecordReader( recMap );
            }
            else if ( sRecords == 2 ) {
                assert padRawValueArray_ != null;
                return new PreviousRecordReader( recMap );
            }
            else {
                throw new CdfFormatException( "Unknown sparse record type "
                                            + sRecords );
            }
        }
    }

    /**
     * Object which can read record values for this variable.
     * This provides the implementations of several of the Variable methods.
     */
    private interface RecordReader {

        /**
         * Indicates whether a real file-based record exists for the given
         * record index.
         *
         * @param  irec  record index
         * @return  true iff a file-based record exists for irec
         */
        boolean hasRecord( int irec );

        /**
         * Reads the data from a single record into a supplied raw value array.
         *
         * @param  irec  record index
         * @param  rawValueArray  workspace array
         */
        void readRawRecord( int irec, Object rawValueArray )
            throws IOException;

        /**
         * Reads the data from a single record and returns it as an object
         * of a suitable type for this variable.
         *
         * @param  irec  record index
         * @param  rowMajor  required majority of output array
         * @param  rawValueArrayWorkspace  workspace array
         * @return   a new object containing shaped result
         */
        Object readShapedRecord( int irec, boolean rowMajor,
                                 Object rawValueArrayWorkspace )
            throws IOException;
    }

    /**
     * RecordReader implementation for non-record-varying variables.
     */
    private class NoVaryRecordReader implements RecordReader {
        private final Object rawValue_;
        private final Object rowMajorValue_;
        private final Object colMajorValue_;

        /**
         * Constructor.
         *
         * @param   recMap  record map
         */
        NoVaryRecordReader( RecordMap recMap ) throws IOException {

            // When record variance is false, the fixed value appears
            // to be located where you would otherwise expect to find record #0.
            // Read it once and store it in raw, row-major and column-major
            // versions for later use.
            RecordReader rt = new UnsparseRecordReader( recMap );
            rawValue_ = createRawValueArray();
            rt.readRawRecord( 0, rawValue_ );
            rowMajorValue_ = shaper_.shape( rawValue_, true );
            colMajorValue_ = shaper_.shape( rawValue_, false );
        }
        public boolean hasRecord( int irec ) {
            return false;
        }
        public void readRawRecord( int irec, Object rawValueArray ) {
            System.arraycopy( rawValue_, 0, rawValueArray, 0, rvaleng_ );
        }
        public Object readShapedRecord( int irec, boolean rowMajor,
                                        Object work ) {
            return rowMajor ? rowMajorValue_ : colMajorValue_;
        }
    }

    /**
     * RecordReader implementation for non-sparse variables.
     */
    private class UnsparseRecordReader implements RecordReader {
        private final RecordMap recMap_;
        private final int nrec_;
        private final Object zeros_;

        /**
         * Constructor.
         *
         * @param  recMap  record map
         */
        UnsparseRecordReader( RecordMap recMap ) {
            recMap_ = recMap;
            nrec_ = vdr_.maxRec + 1;
            zeros_ = createRawValueArray();
        }
        public boolean hasRecord( int irec ) {
            return irec < nrec_;
        }
        public void readRawRecord( int irec, Object rawValueArray )
                throws IOException {
            if ( hasRecord( irec ) ) {
                int ient = recMap_.getEntryIndex( irec );
                dataReader_.readValue( recMap_.getBuf( ient ),
                                       recMap_.getOffset( ient, irec ),
                                       rawValueArray );
            }
            else {
                System.arraycopy( zeros_, 0, rawValueArray, 0, rvaleng_ );
            }
        }
        public Object readShapedRecord( int irec, boolean rowMajor,
                                        Object work )
                throws IOException {
            if ( hasRecord( irec ) ) {
                int ient = recMap_.getEntryIndex( irec );
                dataReader_.readValue( recMap_.getBuf( ient ),
                                       recMap_.getOffset( ient, irec ),
                                       work );
                return shaper_.shape( work, rowMajor );
            }
            else {
                return null;
            }
        }
    }

    /**
     * RecordReader implementation for record-varying variables
     * with sparse padding or no padding.
     */
    private class PadRecordReader implements RecordReader {
        private final RecordMap recMap_;

        /**
         * Constructor.
         *
         * @param  recMap  record map
         */
        PadRecordReader( RecordMap recMap ) {
            recMap_ = recMap;
        }
        public boolean hasRecord( int irec ) {
            return hasRecord( irec, recMap_.getEntryIndex( irec ) );
        }
        public void readRawRecord( int irec, Object rawValueArray )
                throws IOException {
            int ient = recMap_.getEntryIndex( irec );
            if ( hasRecord( irec, ient ) ) {
                dataReader_.readValue( recMap_.getBuf( ient ),
                                       recMap_.getOffset( ient, irec ),
                                       rawValueArray );
            }
            else {
                System.arraycopy( padRawValueArray_, 0, rawValueArray, 0,
                                  rvaleng_ );
            }
        }
        public Object readShapedRecord( int irec, boolean rowMajor,
                                        Object work )
                throws IOException {
            int ient = recMap_.getEntryIndex( irec );
            if ( hasRecord( irec, ient ) ) {
                dataReader_.readValue( recMap_.getBuf( ient ),
                                       recMap_.getOffset( ient, irec ),
                                       work );
                return shaper_.shape( work, rowMajor );
            }
            else {
                return rowMajor ? shapedPadValueRowMajor_
                                : shapedPadValueColumnMajor_;
            }
        }
        private boolean hasRecord( int irec, int ient ) {
            return ient >= 0 && ient < recMap_.getEntryCount()
                && irec < getRecordCount();
        }
    }

    /**
     * RecordReader implementation for record-varying variables
     * with previous padding.
     */
    private class PreviousRecordReader implements RecordReader {
        private final RecordMap recMap_;

        /**
         * Constructor.
         *
         * @param  recMap  record map
         */
        PreviousRecordReader( RecordMap recMap ) {
            recMap_ = recMap;
        }
        public boolean hasRecord( int irec ) {
            // I'm not sure whether the constraint on getRecordCount ought
            // to be applied here - maybe for previous padding, non-existent
            // records are OK??
            return recMap_.getEntryIndex( irec ) >= 0
                && irec < getRecordCount();
        }
        public void readRawRecord( int irec, Object rawValueArray )
                throws IOException {
            int ient = recMap_.getEntryIndex( irec );
            if ( ient >= 0 ) {
                dataReader_.readValue( recMap_.getBuf( ient ),
                                       recMap_.getOffset( ient, irec ),
                                       rawValueArray );
            }
            else if ( ient == -1 ) {
                System.arraycopy( padRawValueArray_, 0, rawValueArray, 0,
                                  rvaleng_ );
            }
            else {
                int iPrevEnt = -ient - 2;
                long offset = recMap_.getFinalOffsetInEntry( iPrevEnt );
                dataReader_.readValue( recMap_.getBuf( iPrevEnt ), offset,
                                       rawValueArray );
            }
        }
        public Object readShapedRecord( int irec, boolean rowMajor,
                                        Object work )
                throws IOException {
            int ient = recMap_.getEntryIndex( irec );
            if ( ient >= 0 ) {
                dataReader_.readValue( recMap_.getBuf( ient ),
                                       recMap_.getOffset( ient, irec ),
                                       work );
                return shaper_.shape( work, rowMajor );
            }
            else if ( ient == -1 ) {
                return rowMajor ? shapedPadValueRowMajor_
                                : shapedPadValueColumnMajor_;
            }
            else {
                int iPrevEnt = -ient - 2;
                long offset = recMap_.getFinalOffsetInEntry( iPrevEnt );
                dataReader_.readValue( recMap_.getBuf( ient ),
                                       recMap_.getOffset( ient, irec ),
                                       work );
                return shaper_.shape( work, rowMajor );
            }
        }
    }
}
