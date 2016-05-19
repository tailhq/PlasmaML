package io.github.mandar2812.PlasmaML.cdf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.Locale;
import java.util.TimeZone;
import java.util.logging.Level;
import java.util.logging.Logger;
import io.github.mandar2812.PlasmaML.PlasmaML;
/**
 * Handles conversions between TT_TIME2000 (TT since J2000.0)
 * and Unix (UTC since 1970-01-01) times.
 * An instance of this class is valid for a certain range of TT2000 dates
 * (one that does not straddle a leap second).
 * To convert between TT_TIME2000 and Unix time, first acquire the
 * right instance of this class for the given time, and then use it
 * for the conversion.
 *
 * <p>An external leap seconds table can be referenced with the
 * {@value #LEAP_FILE_ENV} environment variable in exactly the same way
 * as for the NASA library.  Otherwise an internal leap seconds table
 * will be used.
 *
 * @author   Mark Taylor
 * @since    8 Aug 2013
 */
public abstract class TtScaler {

    private final double fixOffset_;
    private final double scaleBase_;
    private final double scaleFactor_;
    private final long fromTt2kMillis_;
    private final long toTt2kMillis_;

    /** Number of milliseconds in a day. */
    private static final double MILLIS_PER_DAY = 1000 * 60 * 60 * 24;

    /** Date of the J2000 epoch as a Modified Julian Date. */
    private static final double J2000_MJD = 51544.5;

    /** Date of the Unix epoch (1970-01-01T00:00:00) as an MJD. */
    private static final double UNIXEPOCH_MJD = 40587.0;

    /** TT is ahead of TAI by approximately 32.184 seconds. */
    private static final double TT_TAI_MILLIS = 32184;

    /** Fixed time zone. */
    private static final TimeZone UTC = TimeZone.getTimeZone( "UTC" );

    /** Date of the J2000 epoch (2000-01-01T12:00:00) as a Unix time. */
    public static final double J2000_UNIXMILLIS = 946728000000.0;

    /**
     * Environment variable to locate external leap seconds file ({@value}).
     * The environment variable name and file format are just the same
     * as for the NASA CDF library.
     */
    public static final String LEAP_FILE_ENV = "CDF_LEAPSECONDSTABLE";

    private static final Logger logger_ =
        Logger.getLogger( TtScaler.class.getName() );

    /**
     * TT2000 coefficients:
     *    year, month (1=Jan), day_of_month (1-based),
     *    fix_offset, scale_base, scale_factor.
     * year month day_of_month:
     * TAI-UTC= fix_offset S + (MJD - scale_base) * scale_factor S
     *
     * <p>Array initialiser lifted from gsfc.nssdc.cdf.util.CDFTT2000
     * source code.  That derives it from
     * http://maia.usno.navy.mil/ser7/tai-utc.dat.
     * See also http://cdf.gsfc.nasa.gov/html/CDFLeapSeconds.txt.
     */
    private static final double[][] LTS = new double[][] {
        { 1960,  1,  1,  1.4178180, 37300.0, 0.0012960 },
        { 1961,  1,  1,  1.4228180, 37300.0, 0.0012960 },
        { 1961,  8,  1,  1.3728180, 37300.0, 0.0012960 },
        { 1962,  1,  1,  1.8458580, 37665.0, 0.0011232 },
        { 1963, 11,  1,  1.9458580, 37665.0, 0.0011232 },
        { 1964,  1,  1,  3.2401300, 38761.0, 0.0012960 },
        { 1964,  4,  1,  3.3401300, 38761.0, 0.0012960 },
        { 1964,  9,  1,  3.4401300, 38761.0, 0.0012960 },
        { 1965,  1,  1,  3.5401300, 38761.0, 0.0012960 },
        { 1965,  3,  1,  3.6401300, 38761.0, 0.0012960 },
        { 1965,  7,  1,  3.7401300, 38761.0, 0.0012960 },
        { 1965,  9,  1,  3.8401300, 38761.0, 0.0012960 },
        { 1966,  1,  1,  4.3131700, 39126.0, 0.0025920 },
        { 1968,  2,  1,  4.2131700, 39126.0, 0.0025920 },
        { 1972,  1,  1, 10.0,           0.0, 0.0       },
        { 1972,  7,  1, 11.0,           0.0, 0.0       },
        { 1973,  1,  1, 12.0,           0.0, 0.0       },
        { 1974,  1,  1, 13.0,           0.0, 0.0       },
        { 1975,  1,  1, 14.0,           0.0, 0.0       },
        { 1976,  1,  1, 15.0,           0.0, 0.0       },
        { 1977,  1,  1, 16.0,           0.0, 0.0       },
        { 1978,  1,  1, 17.0,           0.0, 0.0       },
        { 1979,  1,  1, 18.0,           0.0, 0.0       },
        { 1980,  1,  1, 19.0,           0.0, 0.0       },
        { 1981,  7,  1, 20.0,           0.0, 0.0       },
        { 1982,  7,  1, 21.0,           0.0, 0.0       },
        { 1983,  7,  1, 22.0,           0.0, 0.0       },
        { 1985,  7,  1, 23.0,           0.0, 0.0       },
        { 1988,  1,  1, 24.0,           0.0, 0.0       },
        { 1990,  1,  1, 25.0,           0.0, 0.0       },
        { 1991,  1,  1, 26.0,           0.0, 0.0       },
        { 1992,  7,  1, 27.0,           0.0, 0.0       },
        { 1993,  7,  1, 28.0,           0.0, 0.0       },
        { 1994,  7,  1, 29.0,           0.0, 0.0       },
        { 1996,  1,  1, 30.0,           0.0, 0.0       },
        { 1997,  7,  1, 31.0,           0.0, 0.0       },
        { 1999,  1,  1, 32.0,           0.0, 0.0       },
        { 2006,  1,  1, 33.0,           0.0, 0.0       },
        { 2009,  1,  1, 34.0,           0.0, 0.0       },
        { 2012,  7,  1, 35.0,           0.0, 0.0       },
        { 2015,  7,  1, 36.0,           0.0, 0.0       },
    };
    private static TtScaler[] ORDERED_INSTANCES;

    /**
     * Constructor.
     *
     * @param   fixOffset  fixed offset of UTC in seconds from TAI
     * @param   scaleBase  MJD base for scaling
     * @param   scaleFactor   factor for scaling
     * @param   fromTt2kMillis  start of validity range
     *                          in TT milliseconds since J2000
     * @param   toTt2kMillis    end of validity range
     *                          in TT milliseconds since J2000
     */
    public TtScaler( double fixOffset, double scaleBase, double scaleFactor,
                     long fromTt2kMillis, long toTt2kMillis ) {
        fixOffset_ = fixOffset;
        scaleBase_ = scaleBase;
        scaleFactor_ = scaleFactor;
        fromTt2kMillis_ = fromTt2kMillis;
        toTt2kMillis_ = toTt2kMillis;
    }

    /**
     * Converts time in milliseconds from TT since J2000 to UTC since 1970
     * for this scaler.
     *
     * @param  tt2kMillis  TT milliseconds since J2000
     * @return  UTC milliseconds since Unix epoch
     */
    public double tt2kToUnixMillis( long tt2kMillis ) {
        return tt2kToUnixMillis( tt2kMillis,
                                 fixOffset_, scaleBase_, scaleFactor_ );
    }

    /**
     * Returns the start of the validity range of this scaler
     * in TT milliseconds since J2000.
     *
     * @return   validity range start
     */
    public long getFromTt2kMillis() {
        return fromTt2kMillis_;
    }

    /**
     * Returns the end of the validity range of this scaler
     * in TT milliseconds since J2000.
     *
     * @return   validity range end
     */
    public long getToTt2kMillis() {
        return toTt2kMillis_;
    }

    /**
     * Assesses validity of this scaler for a given time.
     * The result will be zero if this scaler is valid,
     * negative if the given time is earlier than this scaler's range, and
     * positive if the given time is later than this scaler's range.
     *
     * @param  tt2kMillis  TT milliseconds since J2000
     * @return  validity signum
     */
    public int compareTt2kMillis( long tt2kMillis ) {
        if ( tt2kMillis < fromTt2kMillis_ ) {
            return -1;
        }
        else if ( tt2kMillis >= toTt2kMillis_ ) {
            return +1;
        }
        else {
            return 0;
        }
    }

    /**
     * Indicates whether and how far a given time is into the duration of
     * a leap second.  If the supplied time falls during a leap second,
     * the number of milliseconds elapsed since the leap second's start
     * is returned.  Otherwise (i.e. nearly always) -1 is returned.
     *
     * @param   tt2kMillis  TT time in milliseconds since J2000
     * @return  a value in the range 0...1000 if in a leap second, otherwise -1
     */
    public abstract int millisIntoLeapSecond( long tt2kMillis );

    /**
     * Searches an ordered array of scaler instances for one that is
     * applicable to a supplied TT time.
     * The supplied array of instances must be ordered and cover the
     * supplied time value; the result of {@link #getTtScalers} is suitable
     * and most likely what you want to use here.
     *
     * @param   tt2kMillis  TT time in milliseconds since J2000
     * @param   orderedScalers  list of TtScaler instances ordered in time
     * @param   i0  initial guess at index of the right answer;
     *              if negative no best guess is assumed
     */
    public static int getScalerIndex( long tt2kMillis,
                                      TtScaler[] orderedScalers, int i0 ) {
        int ns = orderedScalers.length;
        return scalerBinarySearch( tt2kMillis, orderedScalers,
                                   i0 >= 0 ? i0 : ns / 2, 0, ns - 1 );
    }

    /**
     * Recursive binary search of an ordered array of scaler instances
     * for one that covers a given point in time.
     *
     * @param   tt2kMillis  TT time in milliseconds since J2000
     * @param   scalers  list of TtScaler instances ordered in time
     * @param   i0  initial guess at index of the right answer
     * @param   imin  minimum possible value of the right answer
     * @parma   imax  maximum possible value of the right answer
     */
    private static int scalerBinarySearch( long tt2kMillis, TtScaler[] scalers,
                                           int i0, int imin, int imax ) {

        // If the guess is correct, return it directly.
        int icmp = scalers[ i0 ].compareTt2kMillis( tt2kMillis );
        if ( icmp == 0 ) {
            return i0;
        }

        // Sanity check.  This condition shouldn't happen, but could do
        // for one of two reasons: a programming error in this code,
        // or an improperly ordered scalers array.
        if ( i0 < imin || i0 > imax ) {
            return -1;
        }
        assert i0 >= imin && i0 <= imax;

        // Bisect up or down and recurse.
        if ( icmp < 0 ) {
            return scalerBinarySearch( tt2kMillis, scalers,
                                       i0 - ( i0 - imin + 1 ) / 2,
                                       imin, i0 - 1 );
        }
        else {
            assert icmp > 0;
            return scalerBinarySearch( tt2kMillis, scalers,
                                       i0 + ( imax - i0 + 1 ) / 2,
                                       i0 + 1, imax );
        }
    }

    /**
     * Converts time in milliseconds from TT since J2000 to UTC since 1970
     * for given coefficients.
     *
     * @param   tt2kMillis  TT milliseconds since J2000
     * @param   fixOffset  fixed offset of UTC in seconds from TAI
     * @param   scaleBase  MJD base for scaling
     * @param   scaleFactor   factor for scaling
     * @return  UTC milliseconds since Unix epoch
     */
    private static double tt2kToUnixMillis( long tt2kMillis, double fixOffset,
                                            double scaleBase,
                                            double scaleFactor ) {
        double mjd = ((double) tt2kMillis) / MILLIS_PER_DAY + J2000_MJD;
        double utcOffsetSec = fixOffset + ( mjd - scaleBase ) * scaleFactor;
        double utcOffsetMillis = utcOffsetSec * 1000;
        return tt2kMillis - TT_TAI_MILLIS - utcOffsetMillis + J2000_UNIXMILLIS;
    }

    /**
     * Converts time in milliseconds from UTC since 1970 to TT since J2000
     * for given coefficients.
     *
     * @param   unixMillis  UTC milliseconds since the Unix epoch
     * @param   fixOffset  fixed offset of UTC in seconds from TAI
     * @param   scaleBase  MJD base for scaling
     * @param   scaleFactor   factor for scaling
     * @return  TT milliseconds since J2000
     */
    private static double unixToTt2kMillis( long unixMillis, double fixOffset,
                                            double scaleBase,
                                            double scaleFactor ) {
        double mjd = ((double) unixMillis) / MILLIS_PER_DAY + UNIXEPOCH_MJD;
        double utcOffsetSec = fixOffset + ( mjd - scaleBase ) * scaleFactor;
        double utcOffsetMillis = utcOffsetSec * 1000;
        return unixMillis + TT_TAI_MILLIS + utcOffsetMillis - J2000_UNIXMILLIS;
    }

    /**
     * Returns an ordered list of scalers covering the whole range of times.
     * Ordering is by time, as per the {@link #compareTt2kMillis} method;
     * every long <code>tt2kMillis</code> value will be valid for one of
     * the list.
     *
     * @return  ordered list of time scalers
     */
    public static synchronized TtScaler[] getTtScalers() {
        if ( ORDERED_INSTANCES == null ) {
            ORDERED_INSTANCES = createTtScalers();
        }
        return ORDERED_INSTANCES.clone();
    }

    /**
     * Creates an ordered list of instances covering the whole range of times.
     *
     * @return  ordered list of time scaler instances
     */
    private static TtScaler[] createTtScalers() {

        // Acquire leap seconds table.
        LtEntry[] ents = readLtEntries();
        int nent = ents.length;
        logger_.config( "CDF Leap second table: " + ents.length + " entries, "
                      + "last is " + ents[ nent - 1 ] );
        List<TtScaler> list = new ArrayList<TtScaler>();

        // Add a scaler valid from the start of time till the first LTS entry.
        // I'm not certain this has the correct formula, but using TT
        // prior to 1960 is a bit questionable in any case.
        LtEntry firstEnt = ents[ 0 ];
        list.add( new NoLeapTtScaler( 0, 0, 0, Long.MIN_VALUE,
                                      firstEnt.getDateTt2kMillis() ) );

        // Add scalers corresponding to each entry in the LTS array except
        // the final one.
        for ( int ie = 0; ie < nent - 1; ie++ ) {
            LtEntry ent0 = ents[ ie ];
            LtEntry ent1 = ents[ ie + 1 ];
            long fromValid = ent0.getDateTt2kMillis();
            long toValid = ent1.getDateTt2kMillis();

            // In case of a leap second, add two: one to cover just the leap
            // second, and another to cover the rest of the range till the
            // next entry starts.
            if ( ent1.hasPrecedingLeapSecond() ) {
                list.add( new NoLeapTtScaler( ent0, fromValid,
                                              toValid - 1000 ) );
                list.add( new LeapDurationTtScaler( ent0, toValid - 1000 ) );
            }

            // In case of no leap second, add a single scaler covering
            // the whole period.
            else {
                list.add( new NoLeapTtScaler( ent0, fromValid, toValid ) );
            }
        }

        // Add a scaler covering the period from the start of the last
        // entry till the end of time.
        LtEntry lastEnt = ents[ nent - 1 ];
        list.add( new NoLeapTtScaler( lastEnt, lastEnt.getDateTt2kMillis(),
                                      Long.MAX_VALUE ) );

        // Return as array.
        return list.toArray( new TtScaler[ 0 ] );
    }

    /**
     * Acquires the table of leap seconds from an internal array or external
     * file as appropriate.
     *
     * @return   leap second entry file
     */
    private static LtEntry[] readLtEntries() {

        // Attempt to read the leap seconds from an external file.
        LtEntry[] fentries = null;
        try {
            fentries = readLtEntriesFile();
        }
        catch ( IOException e ) {
            logger_.log( Level.WARNING,
                         "Failed to read external leap seconds file: " + e, e );
        }
        catch ( RuntimeException e ) {
            logger_.log( Level.WARNING,
                         "Failed to read external leap seconds file: " + e, e );
        }
        if ( fentries != null ) {
            return fentries;
        }

        // If that doesn't work, use the internal hard-coded table.
        else {
            logger_.config( "Using internal leap seconds table" );
            int nent = LTS.length;
            LtEntry[] entries = new LtEntry[ nent ];
            for ( int i = 0; i < nent; i++ ) {
                entries[ i ] = new LtEntry( LTS[ i ] );
            }
            return entries;
        }
    }

    /**
     * Attempts to read the leap seconds table from an external file.
     * As per the NASA library, this is pointed at by an environment variable.
     *
     * @return  leap seconds table, or null if not found
     */
    private static LtEntry[] readLtEntriesFile() throws IOException {
        String ltLoc;
        try {
            ltLoc = PlasmaML.leapSecondsFile();
        }
        catch ( SecurityException e ) {
            logger_.config( "Can't access external leap seconds file: " + e );
            return null;
        }
        if ( ltLoc == null ) {
            return null;
        }
        logger_.config( "Reading leap seconds from file " + ltLoc );
        File file = new File( ltLoc );
        BufferedReader in = new BufferedReader( new FileReader( file ) );
        List<LtEntry> list = new ArrayList<LtEntry>();
        for ( String line; ( line = in.readLine() ) != null; ) {
            if ( ! line.startsWith( ";" ) ) {
                String[] fields = line.trim().split( "\\s+" );
                if ( fields.length != 6 ) {
                    throw new IOException( "Bad leap second file format - got "
                                         + fields.length + " fields not 6"
                                         + " at line \"" + line + "\"" );
                }
                try {
                    int year = Integer.parseInt( fields[ 0 ] );
                    int month = Integer.parseInt( fields[ 1 ] );
                    int dom = Integer.parseInt( fields[ 2 ] );
                    double fixOffset = Double.parseDouble( fields[ 3 ] );
                    double scaleBase = Double.parseDouble( fields[ 4 ] );
                    double scaleFactor = Double.parseDouble( fields[ 5 ] );
                    list.add( new LtEntry( year, month, dom, fixOffset,
                                           scaleBase, scaleFactor ) );
                }
                catch ( NumberFormatException e ) {
                    throw (IOException)
                          new IOException( "Bad entry in leap seconds file" )
                         .initCause( e );
                }
            }
        }
        return list.toArray( new LtEntry[ 0 ] );
    }

    /**
     * TtScaler implementation which does not contain any leap seconds.
     */
    private static class NoLeapTtScaler extends TtScaler {

        /**
         * Constructs a NoLeapScaler from coefficients.
         *
         * @param   fixOffset  fixed offset of UTC in seconds from TAI
         * @param   scaleBase  MJD base for scaling
         * @param   scaleFactor   factor for scaling
         * @param   fromTt2kMillis  start of validity range
         *                          in TT milliseconds since J2000
         * @param   toTt2kMillis    end of validity range
         *                          in TT milliseconds since J2000
         */
        NoLeapTtScaler( double fixOffset, double scaleBase, double scaleFactor,
                        long fromTt2kMillis, long toTt2kMillis ) {
            super( fixOffset, scaleBase, scaleFactor,
                   fromTt2kMillis, toTt2kMillis );
        }

        /**
         * Constructs a NoLeapTtScaler from an LtEntry.
         *
         * @param  ltEnt   LTS table entry object
         * @param   fromTt2kMillis  start of validity range
         *                          in TT milliseconds since J2000
         * @param   toTt2kMillis    end of validity range
         *                          in TT milliseconds since J2000
         */
        NoLeapTtScaler( LtEntry ltEnt,
                        long fromTt2kMillis, long toTt2kMillis ) {
            this( ltEnt.fixOffset_, ltEnt.scaleBase_, ltEnt.scaleFactor_,
                  fromTt2kMillis, toTt2kMillis );
        }

        public int millisIntoLeapSecond( long tt2kMillis ) {
            return -1;
        }
    }

    /**
     * TtScaler implementation whose whole duration represents a single
     * positive leap second.
     */
    private static class LeapDurationTtScaler extends TtScaler {
        private final long leapStartTt2kMillis_;

        /**
         * Constructor.
         *
         * @param  ltEnt   LTS table entry object
         * @param  leapStartTt2kMillis  start of leap second (hence validity
         *                              range) in TT milliseconds since J2000
         */
        LeapDurationTtScaler( LtEntry ltEnt, long leapStartTt2kMillis ) {
            super( ltEnt.fixOffset_, ltEnt.scaleBase_, ltEnt.scaleFactor_,
                   leapStartTt2kMillis, leapStartTt2kMillis + 1000 );
            leapStartTt2kMillis_ = leapStartTt2kMillis;
        }

        public int millisIntoLeapSecond( long tt2kMillis ) {
            long posdiff = tt2kMillis - leapStartTt2kMillis_;
            return posdiff >= 0 && posdiff <= 1000 ? (int) posdiff : -1;
        }
    }

    /**
     * Represents one entry in the LTS array corresponding to leap second
     * ranges.
     */
    private static class LtEntry {
        final int year_;
        final int month_;
        final int dom_;
        final double fixOffset_;
        final double scaleBase_;
        final double scaleFactor_;

        /**
         * Constructs entry from enumerated coefficients.
         *
         * @param   year   leap second year AD
         * @param   month  leap second month (1-based)
         * @param   dom    leap second day of month (1-based)
         * @param   fixOffset  fixed offset of UTC in seconds from TAI
         * @param   scaleBase  MJD base for scaling
         * @param   scaleFactor   factor for scaling
         */
        public LtEntry( int year, int month, int dom, double fixOffset,
                        double scaleBase, double scaleFactor ) {
            year_ = year;
            month_ = month;
            dom_ = dom;
            fixOffset_ = fixOffset;
            scaleBase_ = scaleBase;
            scaleFactor_ = scaleFactor;
        }

        /**
         * Constructs entry from array of 6 doubles.
         *
         * @param   ltCoeffs  6-element array of coefficients from LTS array:
         *                    year, month, dom, offset, base, factor
         */
        public LtEntry( double[] ltCoeffs ) {
             this( (int) ltCoeffs[ 0 ],
                   (int) ltCoeffs[ 1 ],
                   (int) ltCoeffs[ 2 ],
                   ltCoeffs[ 3 ],
                   ltCoeffs[ 4 ],
                   ltCoeffs[ 5 ] );
             assert year_ == ltCoeffs[ 0 ];
             assert month_ == ltCoeffs[ 1 ];
             assert dom_ == ltCoeffs[ 2 ];
        }

        /**
         * Returns the number of milliseconds in TT since J2000 corresponding
         * to the date associated with this entry.
         *
         * @return   TT millis since J2000
         */
        public long getDateTt2kMillis() {
             GregorianCalendar gcal = new GregorianCalendar( UTC, Locale.UK );
             gcal.clear();
             gcal.set( year_, month_ - 1, dom_ );
             long unixMillis = gcal.getTimeInMillis();
             return (long) unixToTt2kMillis( unixMillis, fixOffset_,
                                             scaleBase_, scaleFactor_ );
        }

        /**
         * Indicates whether there is a single positive leap second
         * immediately preceding the date associated with this entry.
         *
         * @return  true iff there is an immediately preceding leap second
         */
        public boolean hasPrecedingLeapSecond() {

            // This implementation is not particularly intuitive or robust,
            // but it's correct for the LTS hard-coded at time of writing,
            // and that array is not likely to undergo changes which would
            // invalidate this algorithm.
            return scaleFactor_ == 0;
        }

        @Override
        public String toString() {
            return year_ + "-" + month_ + "-" + dom_ + ": "
                 + fixOffset_ + ", " + scaleBase_ + ", " + scaleFactor_;
        }
    }
}
