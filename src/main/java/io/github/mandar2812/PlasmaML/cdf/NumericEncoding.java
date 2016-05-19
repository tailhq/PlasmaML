package io.github.mandar2812.PlasmaML.cdf;

/**
 * Enumeration of numeric encoding values supported by CDF.
 *
 * @author   Mark Taylor
 * @since    20 Jun 2013
 */
public enum NumericEncoding {

    NETWORK( Boolean.TRUE ),
    SUN( Boolean.TRUE ),
    NeXT( Boolean.TRUE ),
    MAC( Boolean.TRUE ),
    HP( Boolean.TRUE ),
    SGi( Boolean.TRUE ),
    IBMRS( Boolean.TRUE ),

    DECSTATION( Boolean.FALSE ),
    IBMPC( Boolean.FALSE ),
    ALPHAOSF1( Boolean.FALSE ),
    ALPHAVMSi( Boolean.FALSE ),

    VAX( null ),
    ALPHAVMSd( null ),
    ALPHAVMSg( null );

    private final Boolean isBigendian_;

    /**
     * Constructor.
     *
     * @param  isBigendian  TRUE for simple big-endian,
     *                      FALSE for simple little-endian,
     *                      null for something else
     */
    NumericEncoding( Boolean isBigendian ) {
        isBigendian_ = isBigendian;
    }

    /**
     * Gives the big/little-endianness of this encoding, if that's all
     * the work that has to be done.
     * If the return value is non-null, then numeric values are
     * encoded the same way that java does it (two's complement for
     * integers and IEEE754 for floating point) with big- or little-endian
     * byte ordering, according to the return value.
     * Otherwise, some unspecified encoding is in operation.
     *
     * @return  TRUE for simple big-endian, FALSE for simple little-endian,
     *          null for something weird
     */
    public Boolean isBigendian() {
        return isBigendian_;
    }

    /**
     * Returns the encoding corresponding to the value of the
     * <code>encoding</code> field of the CDF Descriptor Record.
     *
     * @param  code  encoding code
     * @return  encoding object
     * @throws  CdfFormatException  if code is unknown
     */
    public static NumericEncoding getEncoding( int code )
            throws CdfFormatException {
        switch ( code ) {
            case  1: return NETWORK;
            case  2: return SUN;
            case  3: return VAX;
            case  4: return DECSTATION;
            case  5: return SGi;
            case  6: return IBMPC;
            case  7: return IBMRS;
            case  9: return MAC;
            case 11: return HP;
            case 12: return NeXT;
            case 13: return ALPHAOSF1;
            case 14: return ALPHAVMSd;
            case 15: return ALPHAVMSg;
            case 16: return ALPHAVMSi;
            default:
                throw new CdfFormatException( "Unknown numeric encoding "
                                            + code );
        }
    }
}
