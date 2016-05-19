package io.github.mandar2812.PlasmaML.cdf.util;

import java.util.logging.ConsoleHandler;
import java.util.logging.Formatter;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Utilities for controlling logging level.
 *
 * @author   Mark Taylor
 * @since    21 Jun 2013
 */
public class LogUtil {

    /**
     * Private constructor prevents instantiation.
     */
    private LogUtil() {
    }

    /**
     * Sets the logging verbosity of the root logger and ensures that
     * logging messages at that level are reported to the console.
     * You'd think this would be simple, but it requires jumping through hoops.
     *
     * @param   verbose  0 for normal, positive for more, negative for less
     *          (0=INFO, +1=CONFIG, -1=WARNING)
     */
    public static void setVerbosity( int verbose ) {

        // Set a level based on the given verbosity.
        int ilevel = Level.INFO.intValue() - ( verbose * 100 );
        Level level = Level.parse( Integer.toString( ilevel ) );

        // Set the root logger's level to this value.
        Logger rootLogger = Logger.getLogger( "" ); 
        rootLogger.setLevel( level );

        // Make sure that the root logger's console handler will actually
        // emit these messages.  By default it seems that anything below
        // INFO is squashed.
        Handler[] rootHandlers = rootLogger.getHandlers();
        if ( rootHandlers.length > 0 &&
             rootHandlers[ 0 ] instanceof ConsoleHandler ) {
            rootHandlers[ 0 ].setLevel( level );
            rootHandlers[ 0 ].setFormatter( new LineFormatter( false ) );
        }
        for ( int i = 0; i < rootHandlers.length; i++ ) {
            rootHandlers[ i ].setLevel( level );
        }
    }

    /**
     * Compact log record formatter.  Unlike the default
     * {@link java.util.logging.SimpleFormatter} this generally uses only
     * a single line for each record.
     */
    public static class LineFormatter extends Formatter {

        private final boolean debug_;

        /**
         * Constructor.
         *
         * @param   debug  iff true, provides more information per log message
         */
        public LineFormatter( boolean debug ) {
            debug_ = debug;
        }

        public String format( LogRecord record ) {
            StringBuffer sbuf = new StringBuffer();
            sbuf.append( record.getLevel().toString() )
                .append( ": " )
                .append( formatMessage( record ) );
            if ( debug_ ) {
                sbuf.append( ' ' )
                    .append( '(' )
                    .append( record.getSourceClassName() )
                    .append( '.' )
                    .append( record.getSourceMethodName() )
                    .append( ')' );
            }
            sbuf.append( '\n' );
            return sbuf.toString();
        }
    }
}
