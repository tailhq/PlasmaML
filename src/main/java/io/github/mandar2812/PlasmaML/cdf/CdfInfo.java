package io.github.mandar2812.PlasmaML.cdf;

/**
 * Encapsulates some global information about a CDF file.
 *
 * @author   Mark Taylor
 * @since    20 Jun 2013
 */
public class CdfInfo {
    private final boolean rowMajor_;
    private final int[] rDimSizes_;
    private final int leapSecondLastUpdated_;

    /**
     * Constructor.
     *
     * @param  rowMajor  true for row majority, false for column majority
     * @param  rDimSizes   array of dimension sizes for rVariables
     * @param  leapSecondLastUpdated  value of the GDR LeapSecondLastUpdated
     *         field
     */
    public CdfInfo( boolean rowMajor, int[] rDimSizes,
                    int leapSecondLastUpdated ) {
        rowMajor_ = rowMajor;
        rDimSizes_ = rDimSizes;
        leapSecondLastUpdated_ = leapSecondLastUpdated;
    }

    /**
     * Indicates majority of CDF arrays.
     *
     * @return  true for row majority, false for column majority
     */
    public boolean getRowMajor() {
        return rowMajor_;
    }

    /**
     * Returns array dimensions for rVariables.
     *
     * @return  array of dimension sizes for rVariables
     */
    public int[] getRDimSizes() {
        return rDimSizes_;
    }

    /**
     * Returns the date of the last leap second the CDF file knows about.
     * This is the value of the LeapSecondLastUpdated field from the GDR
     * (introduced at CDF v3.6).  The value is an integer whose
     * decimal representation is of the form YYYYMMDD.
     * Values 0 and -1 have special meaning (no last leap second).
     *
     * @return   last known leap second indicator
     */
    public int getLeapSecondLastUpdated() {
        return leapSecondLastUpdated_;
    }
}
