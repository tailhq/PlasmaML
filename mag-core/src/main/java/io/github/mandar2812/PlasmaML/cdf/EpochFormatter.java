package io.github.mandar2812.PlasmaML.cdf;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.Locale;
import java.util.TimeZone;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Does string formatting of epoch values in various representations.
 * The methods of this object are not in general thread-safe.
 *
 * @author   Mark Taylor
 * @since    21 Jun 2013
 */
public class EpochFormatter {

    private final DateFormat epochMilliFormat_ =
        createDateFormat( "yyyy-MM-dd'T'HH:mm:ss.SSS" );
    private final DateFormat epochSecFormat_ =
        createDateFormat( "yyyy-MM-dd'T'HH:mm:ss" );
    private final int iMaxValidTtScaler_;
    private int iLastTtScaler_ = -1;

    private static final TimeZone UTC = TimeZone.getTimeZone( "UTC" );
    private static final long HALF_DAY = 1000 * 60 * 60 * 12;
    private static final TtScaler[] TT_SCALERS = TtScaler.getTtScalers();
    private static final long LAST_KNOWN_LEAP_UNIX_MILLIS =
        getLastKnownLeapUnixMillis( TT_SCALERS );
    private static final Logger logger_ =
        Logger.getLogger( EpochFormatter.class.getName() );

    /**
     * Configures behaviour when a date is encountered which is known to
     * have incorrectly applied leap seconds.
     * If true, a RuntimeException is thrown, if false a log message is written.
     */
    public static boolean FAIL_ON_LEAP_ERROR = false;

    /** 0 A.D. in Unix milliseconds as used by EPOCH/EPOCH16 data types. */
    public static final long AD0_UNIX_MILLIS = getAd0UnixMillis();

    /**
     * Constructs a formatter without leap second awareness.
     */
    public EpochFormatter() {
        this( 0 );
    }

    /**
     * Constructs a formatter aware of the latest known leap second.
     *
     * @param  leapSecondLastUpdated  value of GDR LeapSecondLastUpdated
     *         field (YYYYMMDD, or -1 for unused, or 0 for no leap seconds)
     */
    public EpochFormatter( int leapSecondLastUpdated ) {
        long lastDataLeapUnixMillis =
            getLastDataLeapUnixMillis( leapSecondLastUpdated );

        /* If we know about leap seconds later than the last known one
         * supplied (presumably acquired from a data file),
         * issue a warning that an update might be a good idea. */
        if ( lastDataLeapUnixMillis > LAST_KNOWN_LEAP_UNIX_MILLIS &&
             lastDataLeapUnixMillis - LAST_KNOWN_LEAP_UNIX_MILLIS > HALF_DAY ) {
            DateFormat fmt = createDateFormat( "yyyy-MM-dd" );
            String msg = new StringBuffer()
               .append( "Data knows more leap seconds than library" )
               .append( " (" )
               .append( fmt.format( new Date( lastDataLeapUnixMillis
                                            + HALF_DAY ) ) )
               .append( " > " )
               .append( fmt.format( new Date( LAST_KNOWN_LEAP_UNIX_MILLIS
                                            + HALF_DAY ) ) )
               .append( ")" )
               .toString();
            logger_.warning( msg );
        }

        /* If the supplied last known leap second is known to be out of date
         * (because we know of a later one), then prepare to complain if
         * this formatter is called upon to perform a conversion of
         * a date that would be affected by leap seconds we know about,
         * but the data file didn't. */
        if ( lastDataLeapUnixMillis > 0 ) {
            long lastDataLeapTt2kMillis =
                lastDataLeapUnixMillis - (long) TtScaler.J2000_UNIXMILLIS;
            iMaxValidTtScaler_ = getScalerIndex( lastDataLeapTt2kMillis );
        }
        else {
            iMaxValidTtScaler_ = TT_SCALERS.length - 1;
        }
    }

    /**
     * Formats a CDF EPOCH value as an ISO-8601 date.
     *
     * @param  epoch  EPOCH value
     * @return   date string
     */
    public String formatEpoch( double epoch ) {
        long unixMillis = (long) ( epoch + AD0_UNIX_MILLIS );
        Date date = new Date( unixMillis );
        return epochMilliFormat_.format( date );
    }

    /**
     * Formats a CDF EPOCH16 value as an ISO-8601 date.
     *
     * @param   epoch1  first element of EPOCH16 pair (seconds since 0AD)
     * @param   epoch2  second element of EPOCH16 pair (additional picoseconds)
     * @return  date string
     */
    public String formatEpoch16( double epoch1, double epoch2 ) {
        long unixMillis = (long) ( epoch1 * 1000 ) + AD0_UNIX_MILLIS;
        Date date = new Date( unixMillis );
        long plusPicos = (long) epoch2;
        if ( plusPicos < 0 || plusPicos >= 1e12 ) {
            return "??";
        }
        String result = new StringBuffer( 32 )
            .append( epochSecFormat_.format( date ) )
            .append( '.' )
            .append( prePadWithZeros( plusPicos, 12 ) )
            .toString();
        assert result.length() == 32;
        return result;
    }

    /**
     * Formats a CDF TIME_TT2000 value as an ISO-8601 date.
     *
     * @param  timeTt2k  TIME_TT2000 value
     * @return  date string
     */
    public String formatTimeTt2000( long timeTt2k ) {

        // Special case - see "Variable Pad Values" section
        // (sec 2.3.20 at v3.4, and footnote) of CDF Users Guide.
        if ( timeTt2k == Long.MIN_VALUE ) {
            return "9999-12-31T23:59:59.999999999";
        }

        // Second special case - not sure if this is documented, but
        // advised by Michael Liu in email to MBT 12 Aug 2013.
        else if ( timeTt2k == Long.MIN_VALUE + 1 ) {
            return "0000-01-01T00:00:00.000000000";
        }

        // Split the raw long value into a millisecond base and
        // nanosecond adjustment.
        long tt2kMillis = timeTt2k / 1000000;
        int plusNanos = (int) ( timeTt2k % 1000000 );
        if ( plusNanos < 0 ) {
            tt2kMillis--;
            plusNanos += 1000000;
        }

        // Get the appropriate TT scaler object for this epoch.
        int scalerIndex = getScalerIndex( tt2kMillis );
        if ( scalerIndex > iMaxValidTtScaler_ ) {
            String msg = new StringBuffer()
               .append( "CDF TIME_TT2000 date formatting failed" )
               .append( " - library leap second table known to be out of date" )
               .append( " with respect to data." )
               .append( " Update " )
               .append( TtScaler.LEAP_FILE_ENV )
               .append( " environment variable to point at file" )
               .append( " http://cdf.gsfc.nasa.gov/html/CDFLeapSeconds.txt" )
               .toString();
            if ( FAIL_ON_LEAP_ERROR ) {
                throw new RuntimeException( msg );
            }
            else {
                logger_.log( Level.SEVERE, msg );
            }
        }
        TtScaler scaler = TT_SCALERS[ scalerIndex ];

        // Use it to convert to Unix time, which is UTC.
        long unixMillis = (long) scaler.tt2kToUnixMillis( tt2kMillis );
        int leapMillis = scaler.millisIntoLeapSecond( tt2kMillis );

        // Format the unix time as an ISO-8601 date.
        // In most (99.999998%) cases this is straightforward.
        final String txt;
        if ( leapMillis < 0 ) {
            Date date = new Date( unixMillis );
            txt = epochMilliFormat_.format( date );
        }

        // However if we happen to fall during a leap second, we have to
        // do some special (and not particularly elegant) handling to
        // produce the right string, since the java DateFormat
        // implementation can't(?) be persuaded to cope with 61 seconds
        // in a minute.
        else {
            Date date = new Date( unixMillis - 1000 );
            txt = epochMilliFormat_.format( date )
                 .replaceFirst( ":59\\.", ":60." );
        }

        // Append the nanoseconds part and return.
        return txt + prePadWithZeros( plusNanos, 6 );
    }

    /**
     * Returns the index into the TT_SCALERS array of the TtScaler
     * instance that is valid for a given time.
     *
     * @param  tt2kMillis  TT time since J2000 in milliseconds
     * @return  index into TT_SCALERS
     */
    private int getScalerIndex( long tt2kMillis ) {

        // Use the most recently used value as the best guess.
        // There's a good chance it's the right one.
        int index = TtScaler
                   .getScalerIndex( tt2kMillis, TT_SCALERS, iLastTtScaler_ );
        iLastTtScaler_ = index;
        return index;
    }

    /**
     * Constructs a DateFormat object for a given pattern for UTC.
     *
     * @param  pattern  formatting pattern
     * @return  format
     * @see   java.text.SimpleDateFormat
     */
    private static DateFormat createDateFormat( String pattern ) {
        DateFormat fmt = new SimpleDateFormat( pattern );
        fmt.setTimeZone( UTC );
        fmt.setCalendar( new GregorianCalendar( UTC, Locale.UK ) );
        return fmt;
    }

    /**
     * Returns the CDF epoch (0000-01-01T00:00:00)
     * in milliseconds since the Unix epoch (1970-01-01T00:00:00).
     *
     * @return  -62,167,219,200,000
     */
    private static long getAd0UnixMillis() {
        GregorianCalendar cal = new GregorianCalendar( UTC, Locale.UK );
        cal.setLenient( true );
        cal.clear();
        cal.set( 0, 0, 1, 0, 0, 0 );
        long ad0 = cal.getTimeInMillis();

        // Fudge factor to make this calculation match the apparent result
        // from the CDF library.  Not quite sure why it's required, but
        // I think something to do with the fact that the first day is day 1
        // and signs around AD0/BC0.
        long fudge = 1000 * 60 * 60 * 24 * 2;  // 2 days
        return ad0 + fudge;
    }

    /**
     * Pads a numeric value with zeros to return a fixed length string
     * representing a given numeric value.
     *
     * @param  value  number
     * @param  leng   number of characters in result
     * @return   leng-character string containing value
     *           padded at start with zeros
     */
    private static String prePadWithZeros( long value, int leng ) {
        String txt = Long.toString( value );
        int nz = leng - txt.length();
        if ( nz == 0 ) {
            return txt;
        }
        else if ( nz < 0 ) {
            throw new IllegalArgumentException();
        }
        else {
            StringBuffer sbuf = new StringBuffer( leng );
            for ( int i = 0; i < nz; i++ ) {
                sbuf.append( '0' );
            }
            sbuf.append( txt );
            return sbuf.toString();
        }
    }

    /**
     * Returns the date, in milliseconds since the Unix epoch,
     * of the last leap second known by the library.
     *
     * @param  scalers  ordered array of all scalers
     * @return   last leap second epoch in unix milliseconds
     */
    private static long getLastKnownLeapUnixMillis( TtScaler[] scalers ) {
        TtScaler lastScaler = scalers[ scalers.length - 1 ];
        return (long)
               lastScaler.tt2kToUnixMillis( lastScaler.getFromTt2kMillis() );
    }

    /**
     * Returns the date, in milliseconds since the Unix epoch,
     * of the last leap second indicated by an integer in the form
     * used by the GDR LeapSecondLastUpdated field.
     * If no definite value is indicated, Long.MIN_VALUE is returned.
     *
     * @param  leapSecondLastUpdated  value of GDR LeapSecondLastUpdated
     *         field (YYYYMMDD, or -1 for unused, or 0 for no leap seconds)
     * @return   last leap second epoch in unix milliseconds,
     *           or very negative value
     */
    private static long getLastDataLeapUnixMillis( int leapSecondLastUpdated ) {
        if ( leapSecondLastUpdated == 0 ) {
            return Long.MIN_VALUE;
        }
        else if ( leapSecondLastUpdated == -1 ) {
            return Long.MIN_VALUE;
        }
        else {
            DateFormat fmt = createDateFormat( "yyyyMMdd" );
            try {
                return fmt.parse( Integer.toString( leapSecondLastUpdated ) )
                          .getTime();
            }
            catch ( ParseException e ) {
                logger_.warning( "leapSecondLastUpdated="
                               + leapSecondLastUpdated
                               + "; not YYYYMMDD" );
                return Long.MIN_VALUE;
            }
        }
    }
}
