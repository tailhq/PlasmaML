package io.github.mandar2812.PlasmaML.cdf;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import io.github.mandar2812.PlasmaML.cdf.util.LogUtil;

public class OtherTest {

    private static boolean assertionsOn_;
    private static boolean triedNasa_;
    private static Method nasaConvMethod_;
    private EpochFormatter epf_ = new EpochFormatter();

    public void testTtScaler() {
        TtScaler[] scalers = TtScaler.getTtScalers();
        int ns = scalers.length;

        // Check scaler list is properly ordered and contigously covers the
        // whole range of times.
        for ( int i = 0; i < ns - 1; i++ ) {
            long from = scalers[ i ].getFromTt2kMillis();
            long to = scalers[ i ].getToTt2kMillis();
            assert from < to;
            assert to == scalers[ i + 1 ].getFromTt2kMillis();
        }
        assert scalers[ 0 ].getFromTt2kMillis() == Long.MIN_VALUE;
        assert scalers[ ns - 1 ].getToTt2kMillis() == Long.MAX_VALUE;

        // Exhaustive test of binary search.
        for ( int i = 0; i < ns; i++ ) {
            TtScaler scaler = scalers[ i ];
            long from = scalers[ i ].getFromTt2kMillis();
            long to = scalers[ i ].getToTt2kMillis();
            long mid = (long) ( 0.5 * from + 0.5 * to ); // careful of overflow
            checkScalerSearch( from, scalers, i );
            checkScalerSearch( to - 1, scalers, i );
            checkScalerSearch( mid, scalers, i );
        }
    }

    private void checkScalerSearch( long tt2kMillis, TtScaler[] scalers,
                                    int iResult ) {
        for ( int i = 0; i < scalers.length; i++ ) {
            assert TtScaler.getScalerIndex( tt2kMillis, scalers, i ) == iResult;
        }
    }

    public void testTtFormatter() {

        // Spot tests.
        assertTt( 284040064183000000L, "2008-12-31T23:59:58.999000000" );
        assertTt( 284040065184000000L, "2008-12-31T23:59:60.000000000" );
        assertTt( 284040066183000000L, "2008-12-31T23:59:60.999000000" );
        assertTt( 284040066183000023L, "2008-12-31T23:59:60.999000023" );
        assertTt( 284040066184000000L, "2009-01-01T00:00:00.000000000" );
        assertTt( 284040066185000000L, "2009-01-01T00:00:00.001000000" );
        assertTt( 284040065307456789L, "2008-12-31T23:59:60.123456789" );

        // Special values.
        assertTt( Long.MIN_VALUE, "9999-12-31T23:59:59.999999999" );
        assertTt( Long.MIN_VALUE + 1, "0000-01-01T00:00:00.000000000" );

        // Systematic tests for all scaler ranges except the last.
        TtScaler[] scalers = TtScaler.getTtScalers();
        int ns = scalers.length;
        for ( int i = 0; i < ns - 1; i++ ) {
            TtScaler scaler = scalers[ i ];
            long from = scalers[ i ].getFromTt2kMillis();
            long to = scalers[ i ].getToTt2kMillis();
            long mid = (long) ( 0.5 * from + 0.5 * to ); // careful of overflow
            checkWithNasa( from );
            checkWithNasa( from + 50 );
            checkWithNasa( from + 333333333 );
            checkWithNasa( to - 1 );
            checkWithNasa( to + 1 );
            checkWithNasa( to - 55555555 );
            checkWithNasa( to + 99999999 );
            checkWithNasa( mid );
        }

        checkWithNasa( Long.MIN_VALUE / 2 );
        checkWithNasa( Long.MAX_VALUE / 2 );
        checkWithNasa( 284040065307456789L );

        // The NASA library v3.4 appeared to be wrong here: it reported
        // a date of 1707-09-22T11:37:39.106448384 for values larger
        // than about 9223370000000000000L.
        // It was fixed at (or maybe before) v3.6.0.4, so we can run
        // this test now.
        checkWithNasa( 9223370000000000000L );
    }

    private void checkWithNasa( long tt2kNanos ) {
        assert epf_.formatTimeTt2000( tt2kNanos )
              .equals( nasaFormatTimeTt2000( tt2kNanos ) )
            : reportFormats( tt2kNanos );
    }

    private void assertTt( long tt2kNanos, String text ) {
        assert text.equals( epf_.formatTimeTt2000( tt2kNanos ) );
    }

    private static String nasaFormatTimeTt2000( long tt2knanos ) {
        if ( ! triedNasa_ ) {
            try {
                Class<?> ttClazz =
                    Class.forName( "gsfc.nssdc.cdf.util.CDFTT2000" );
                nasaConvMethod_ =
                    ttClazz.getMethod( "toUTCstring", long.class );
            }
            catch ( Throwable e ) {
                System.err.println( "No NASA implementation available:" );
                e.printStackTrace( System.err );
                nasaConvMethod_ = null;
            }

            // Call this method once.  If the native library is not present
            // it fails in the static initialisation, then subsequent calls
            // seem to be OK, but give the wrong result.  So make sure
            // it doesn't run at all in case of initialisation failure.
            try {
                nasaConvMethod_.invoke( null, 0L );
            }
            catch ( Throwable e ) {
                System.err.println( "No NASA implementation available:" );
                e.printStackTrace( System.err );
                nasaConvMethod_ = null;
            }
            triedNasa_ = true;
        }
        if ( nasaConvMethod_ == null ) {
            return "[No NASA CDF library]";
        }
        else {
            try {
                return (String) nasaConvMethod_.invoke( null, tt2knanos );
            }
            catch ( Throwable e ) {
                return "[toUTCstring error: " + e + "]";
            }
        }
    }

    private static boolean checkAssertions() {
        assertionsOn_ = true;
        return true;
    }

    private static void runTests() {
            assert checkAssertions();
        if ( ! assertionsOn_ ) {
            throw new RuntimeException( "Assertions disabled - bit pointless" );
        }
        OtherTest test = new OtherTest();
        test.testTtScaler();
        test.testTtFormatter();
    }

    private static String reportFormats( long tt2kNanos ) {
        return new StringBuffer()
              .append( "nanos: " )
              .append( tt2kNanos )
              .append( "\n\t" )
              .append( "NASA: " )
              .append( nasaFormatTimeTt2000( tt2kNanos ) )
              .append( "\n\t" )
              .append( "JCDF: " )
              .append( new EpochFormatter().formatTimeTt2000( tt2kNanos ) )
              .toString();
    }

    /**
     * Main method.  If run with no arguments runs test.
     * Tests are made using java assertions, so this test must be
     * run with java assertions enabled.  If it's not, it will fail anyway.
     *
     * <p>If run with arguments a utility function that reports JCDF and NASA
     * formatting for TIME_TT2000 nanosecond values.
     *
     */
    public static void main( String[] args ) {

        List<String> argList = new ArrayList<String>( Arrays.asList( args ) );
        int verb = 0;
        for ( Iterator<String> it = argList.iterator(); it.hasNext(); ) {
            String arg = it.next();
            if ( arg.startsWith( "-v" ) ) {
                it.remove();
                verb++;
            }
            else if ( arg.startsWith( "+v" ) ) {
                it.remove();
                verb--;
            }
        }
        LogUtil.setVerbosity( verb );

        // Special case - utility function to report TIME_TT2000 values
        // from the command line.
        if ( argList.size() > 0 ) {
            for ( String arg : argList ) {
                System.out.println( reportFormats( Long.parseLong( arg ) ) );
                System.out.println();
            }
        }

        // Otherwise run tests.
        else {
            runTests();
        }
    }
}
