package io.github.mandar2812.PlasmaML.cdf;

import java.io.IOException;

/**
 * Exception thrown during CDF parsing when the data stream appears either
 * to be in contravention of the CDF format, or uses some feature of
 * the CDF format which is unsupported by the current implementation.
 *
 * @author   Mark Taylor
 * @since    18 Jun 2013
 */
public class CdfFormatException extends IOException {

    /**
     * Constructs an exception with a message.
     *
     * @param  msg  message
     */
    public CdfFormatException( String msg ) {
        super( msg );
    }

    /**
     * Constructs an exception with a message and a cause.
     *
     * @param  msg  message
     * @param  cause   upstream exception
     */
    public CdfFormatException( String msg, Throwable cause ) {
        super( msg );
        initCause( cause );
    }
}
