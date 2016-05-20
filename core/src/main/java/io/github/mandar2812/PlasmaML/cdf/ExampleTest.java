package io.github.mandar2812.PlasmaML.cdf;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import io.github.mandar2812.PlasmaML.cdf.record.VariableAttribute;
import io.github.mandar2812.PlasmaML.cdf.record.Variable;

/**
 * Tests the contents of two of the example files
 * (example1.cdf and example2.cdf) from the samples directory of the
 * CDF distribution.  The assertions in this file were written by
 * examining the output of cdfdump by eye.
 */
public class ExampleTest {

    private static boolean assertionsOn_;

    public void testExample1( File ex1file ) throws IOException {
        CdfContent content = new CdfContent( new CdfReader( ex1file ) );

        GlobalAttribute[] gatts = content.getGlobalAttributes();
        assert gatts.length == 1;
        GlobalAttribute gatt0 = gatts[ 0 ];
        assert "TITLE".equals( gatt0.getName() );
        assert Arrays.equals( new String[] { "CDF title", "Author: CDF" },
                              getEntryShapedValues( gatt0.getEntries() ) );

        VariableAttribute[] vatts = content.getVariableAttributes();
        assert vatts.length == 2;
        assert "FIELDNAM".equals( vatts[ 0 ].getName() );
        assert "UNITS".equals( vatts[ 1 ].getName() );

        Variable[] vars = content.getVariables();
        assert vars.length == 3;
        assert "Time".equals( vars[ 0 ].getName() );
        assert "Latitude".equals( vars[ 1 ].getName() );
        assert "Image".equals( vars[ 2 ].getName() );
        assert vars[ 0 ].getSummary().matches( "INT4 .* 0:\\[\\] T/" );
        assert vars[ 1 ].getSummary().matches( "INT2 .* 1:\\[181\\] T/T" );
        assert vars[ 2 ].getSummary().matches( "INT4 .* 2:\\[10,20\\] T/TT" );

        assert vatts[ 1 ].getEntry( vars[ 0 ] ).getShapedValue()
                                               .equals( "Hour/Minute" );
        assert vatts[ 1 ].getEntry( vars[ 1 ] ) == null;

        assert readShapedRecord( vars[ 0 ], 0, true )
              .equals( new Integer( 23 ) );
        assert readShapedRecord( vars[ 0 ], 1, true ) == null;
        assert readShapedRecord( vars[ 0 ], 2, true ) == null;
        assert Arrays.equals( (short[]) readShapedRecord( vars[ 1 ], 0, true ),
                              shortSequence( -90, 1, 181 ) );
        assert Arrays.equals( (short[]) readShapedRecord( vars[ 1 ], 0, false ),
                              shortSequence( -90, 1, 181 ) );
        assert readShapedRecord( vars[ 1 ], 1, true ) == null;
        assert readShapedRecord( vars[ 1 ], 2, false ) == null;
        assert Arrays.equals( (int[]) readShapedRecord( vars[ 2 ], 0, true ),
                              intSequence( 0, 1, 200 ) );
        assert Arrays.equals( (int[]) readShapedRecord( vars[ 2 ], 1, true ),
                              intSequence( 200, 1, 200 ) );
        assert Arrays.equals( (int[]) readShapedRecord( vars[ 2 ], 2, true ),
                              intSequence( 400, 1, 200 ) );
        int[] sideways = (int[]) readShapedRecord( vars[ 2 ], 0, false );
        assert sideways[ 0 ] == 0;
        assert sideways[ 1 ] == 20;
        assert sideways[ 2 ] == 40;
        assert sideways[ 10 ] == 1;
        assert sideways[ 199 ] == 199;
    }

    public void testExample2( File ex2file ) throws IOException {
        CdfContent content = new CdfContent( new CdfReader( ex2file ) );

        GlobalAttribute[] gatts = content.getGlobalAttributes();
        assert gatts.length == 1;
        GlobalAttribute gatt0 = gatts[ 0 ];
        assert "TITLE".equals( gatt0.getName() );
        assert "An example CDF (2)."
              .equals( ((String) gatt0.getEntries()[ 0 ].getShapedValue())
                      .trim() );

        VariableAttribute[] vatts = content.getVariableAttributes();
        assert vatts.length == 9;
        VariableAttribute fnVatt = vatts[ 0 ];
        VariableAttribute vminVatt = vatts[ 1 ];
        VariableAttribute vmaxVatt = vatts[ 2 ];
        assert fnVatt.getName().equals( "FIELDNAM" );
        assert vminVatt.getName().equals( "VALIDMIN" );
        assert vmaxVatt.getName().equals( "VALIDMAX" );

        Variable[] vars = content.getVariables();
        assert vars.length == 4;
        Variable timeVar = vars[ 0 ];
        Variable lonVar = vars[ 1 ];
        Variable latVar = vars[ 2 ];
        Variable tempVar = vars[ 3 ];
        assert timeVar.getName().equals( "Time" );
        assert lonVar.getName().equals( "Longitude" );
        assert latVar.getName().equals( "Latitude" );
        assert tempVar.getName().equals( "Temperature" );

        assert timeVar.getSummary().matches( "INT4 .* 0:\\[\\] T/" );
        assert lonVar.getSummary().matches( "REAL4 .* 1:\\[2\\] F/T" );
        assert latVar.getSummary().matches( "REAL4 .* 1:\\[2\\] F/T" );
        assert tempVar.getSummary().matches( "REAL4 .* 2:\\[2,2\\] T/TT" );
        assert timeVar.getRecordCount() == 24;
        assert tempVar.getRecordCount() == 24;
        assert lonVar.getRecordCount() == 1;
        assert latVar.getRecordCount() == 1;

        assert ((String) fnVatt.getEntry( timeVar ).getShapedValue()).trim()
                     .equals( "Time of observation" );
        assert vminVatt.getEntry( timeVar ).getShapedValue()
                                           .equals( new Integer( 0 ) );
        assert vmaxVatt.getEntry( timeVar ).getShapedValue()
                                           .equals( new Integer( 2359 ) );
        assert vminVatt.getEntry( lonVar ).getShapedValue()
                                          .equals( new Float( -180f ) );
        assert vmaxVatt.getEntry( lonVar ).getShapedValue()
                                          .equals( new Float( 180f ) );

        assert readShapedRecord( timeVar, 0, true )
              .equals( new Integer( 0 ) );
        assert readShapedRecord( timeVar, 23, false )
              .equals( new Integer( 2300 ) );

        float[] lonVal = new float[] { -165f, -150f };
        float[] latVal = new float[] { 40f, 30f };
        for ( int irec = 0; irec < 24; irec++ ) {
            assert Arrays.equals( (float[]) readShapedRecord( lonVar, irec,
                                                              true ),
                                  lonVal );
            assert Arrays.equals( (float[]) readShapedRecord( latVar, irec,
                                                              false ),
                                  latVal );
        }
        assert Arrays.equals( (float[]) readShapedRecord( tempVar, 0, true ),
                              new float[] { 20f, 21.7f, 19.2f, 20.7f } );
        assert Arrays.equals( (float[]) readShapedRecord( tempVar, 23, true ),
                              new float[] { 21f, 19.5f, 18.4f, 22f } );
        assert Arrays.equals( (float[]) readShapedRecord( tempVar, 23, false ),
                              new float[] { 21f, 18.4f, 19.5f, 22f } );

    }

    public void testTest( File testFile ) throws IOException {
        CdfContent content = new CdfContent( new CdfReader( testFile ) );

        GlobalAttribute[] gatts = content.getGlobalAttributes();
        assert gatts.length == 5;
        assert "Project".equals( gatts[ 0 ].getName() );
        GlobalAttribute gatt1 = gatts[ 1 ];
        assert "PI".equals( gatt1.getName() );
        assert Arrays.equals( new String[] { null, null, null, "Ernie Els" },
                              getEntryShapedValues( gatt1.getEntries() ) );
        GlobalAttribute gatt2 = gatts[ 2 ];
        assert "Test".equals( gatt2.getName() );
        AttributeEntry[] tents = gatt2.getEntries();
        assert tents[ 0 ].getShapedValue().equals( new Double( 5.3432 ) );
        assert tents[ 1 ] == null;
        assert tents[ 2 ].getShapedValue().equals( new Float( 5.5f ) );
        assert Arrays.equals( (float[]) tents[ 3 ].getShapedValue(),
                              new float[] { 5.5f, 10.2f } );
        assert Arrays.equals( (float[]) tents[ 3 ].getRawValue(),
                              new float[] { 5.5f, 10.2f } );
        assert ((Byte) tents[ 4 ].getShapedValue()).byteValue() == 1;
        assert Arrays.equals( (byte[]) tents[ 5 ].getShapedValue(),
                              new byte[] { (byte) 1, (byte) 2, (byte) 3 } );
        assert ((Short) tents[ 6 ].getShapedValue()).shortValue() == -32768;
        assert Arrays.equals( (short[]) tents[ 7 ].getShapedValue(),
                              new short[] { (short) 1, (short) 2 } );
        assert ((Integer) tents[ 8 ].getShapedValue()).intValue() == 3;
        assert Arrays.equals( (int[]) tents[ 9 ].getShapedValue(),
                              new int[] { 4, 5 } );
        assert "This is a string".equals( tents[ 10 ].getShapedValue() );
        assert ((Long) tents[ 11 ].getShapedValue()).longValue() == 4294967295L;
        assert Arrays.equals( (long[]) tents[ 12 ].getShapedValue(),
                              new long[] { 4294967295L, 2147483648L } );
        assert ((Integer) tents[ 13 ].getShapedValue()).intValue() == 65535;
        assert Arrays.equals( (int[]) tents[ 14 ].getShapedValue(),
                              new int[] { 65535, 65534 } );
        assert ((Short) tents[ 15 ].getShapedValue()).shortValue() == 255;
        assert Arrays.equals( (short[]) tents[ 16 ].getShapedValue(),
                              new short[] { 255, 254 } );

        EpochFormatter epf = new EpochFormatter();
        GlobalAttribute gatt3 = gatts[ 3 ];
        assert "TestDate".equals( gatt3.getName() );
        assert "2002-04-25T00:00:00.000"
              .equals( epf
                      .formatEpoch( ((Double)
                                     gatt3.getEntries()[ 1 ].getShapedValue())
                                    .doubleValue() ) );
        assert "2008-02-04T06:08:10.012014016"
              .equals( epf
                      .formatTimeTt2000( ((Long) gatt3.getEntries()[ 2 ]
                                                      .getShapedValue())
                                         .longValue() ) );
        double[] epDate = (double[])
                          gatts[ 4 ].getEntries()[ 0 ].getShapedValue();
        assert "2004-05-13T15:08:11.022033044055"
              .equals( epf.formatEpoch16( epDate[ 0 ], epDate[ 1 ] ) );

        Variable[] vars = content.getVariables();
        Variable latVar = vars[ 0 ];
        assert "Latitude".equals( latVar.getName() );
        assert Arrays.equals( new byte[] { (byte) 1, (byte) 2, (byte) 3 },
                              (byte[]) readShapedRecord( latVar, 0, true ) );
        assert Arrays.equals( new byte[] { (byte) 1, (byte) 2, (byte) 3 },
                              (byte[]) readShapedRecord( latVar, 100, true ) );

        Variable lat1Var = vars[ 1 ];
        assert "Latitude1".equals( lat1Var.getName() );
        assert Arrays.equals( new short[] { (short) 100, (short) 128,
                                            (short) 255 },
                              (short[]) readShapedRecord( lat1Var, 2, true ) );

        Variable longVar = vars[ 2 ];
        assert "Longitude".equals( longVar.getName() );
        assert Arrays.equals( new short[] { (short) 100, (short) 200,
                                            (short) 300 },
                              (short[]) readShapedRecord( longVar, 0, true ) );
        assert Arrays.equals( new short[] { (short) -99, (short) -99,
                                            (short) -99 },
                              (short[]) readShapedRecord( longVar, 1, true ) );

        Variable nameVar = vars[ 8 ];
        assert "Name".equals( nameVar.getName() );
        assert Arrays.equals( new String[] { "123456789 ", "13579     " },
                              (String[]) readShapedRecord( nameVar, 0, true ) );

        Variable tempVar = vars[ 9 ];
        assert "Temp".equals( tempVar.getName() );
        assert Arrays.equals( new float[] { 55.5f, 0f, 66.6f },
                              (float[]) readShapedRecord( tempVar, 0, true ) );
        assert Arrays.equals( new float[] { 0f, 0f, 0f },
                              (float[]) readShapedRecord( tempVar, 1, true ) );

        Variable epVar = vars[ 15 ];
        assert "ep".equals( epVar.getName() );
        assert "1999-03-05T05:06:07.100"
              .equals( epf
                      .formatEpoch( (Double) readShapedRecord( epVar, 0 ) ) );

        Variable ep16Var = vars[ 16 ];
        assert "ep16".equals( ep16Var.getName() );
        double[] ep2 = (double[]) readShapedRecord( ep16Var, 1, true );
        assert "2004-12-29T16:56:24.031411522634"
              .equals( epf.formatEpoch16( ep2[ 0 ], ep2[ 1 ] ) );

        Variable ttVar = vars[ 18 ];
        assert "tt2000".equals( ttVar.getName() );
        assert "2008-12-31T23:59:58.123456789"
              .equals( epf.formatTimeTt2000( (Long)
                                             readShapedRecord( ttVar, 0 ) ) );
        assert "2008-12-31T23:59:60.123456789"
              .equals( epf.formatTimeTt2000( (Long)
                                             readShapedRecord( ttVar, 2 ) ) );
        assert "2009-01-01T00:00:00.123456789"
              .equals( epf.formatTimeTt2000( (Long)
                                             readShapedRecord( ttVar, 3 ) ) );
    }

    private Object readShapedRecord( Variable var, int irec, boolean rowMajor )
            throws IOException {
        return var.readShapedRecord( irec, rowMajor,
                                     var.createRawValueArray() );
    }

    private Object readShapedRecord( Variable var, int irec )
            throws IOException {
        return readShapedRecord( var, irec, true );
    }

    private short[] shortSequence( int start, int step, int count ) {
        short[] array = new short[ count ];
        for ( int i = 0; i < count; i++ ) {
            array[ i ] = (short) ( start + i * step );
        }
        return array;
    }

    private int[] intSequence( int start, int step, int count ) {
        int[] array = new int[ count ];
        for ( int i = 0; i < count; i++ ) {
            array[ i ] = start + i * step;
        }
        return array;
    }

    private static Object[] getEntryShapedValues( AttributeEntry[] entries ) {
        int nent = entries.length;
        Object[] vals = new Object[ nent ];
        for ( int ie = 0; ie < nent; ie++ ) {
            AttributeEntry entry = entries[ ie ];
            vals[ ie ] = entry == null ? null : entry.getShapedValue();
        }
        return vals;
    }

    private static boolean checkAssertions() {
        assertionsOn_ = true;
        return true;
    }


    /**
     * Main method.  Run with locations of the following files as arguments:
     *    cdf34_1-dist/samples/example1.cdf
     *    cdf34_1-dist/samples/example2.cdf
     *    cdf34_1-dist/cdfjava/examples/test.cdf
     * as arguments.  Use -help for help.
     * Tests are made using java assertions, so this test must be
     * run with java assertions enabled.  If it's not, it will fail anyway.
     */
    public static void main( String[] args ) throws IOException {
        assert checkAssertions();
        if ( ! assertionsOn_ ) {
            throw new RuntimeException( "Assertions disabled - bit pointless" );
        }
        String usage = "Usage: " + ExampleTest.class.getName()
                     + " example1.cdf example2.cdf";
        if ( args.length != 3 ) {
            System.err.println( usage );
            System.exit( 1 );
        }
        File ex1 = new File( args[ 0 ] );
        File ex2 = new File( args[ 1 ] );
        File test = new File( args[ 2 ] );
        if ( ! ex1.canRead() || ! ex2.canRead() || ! test.canRead() ) {
            System.err.println( usage );
            System.exit( 1 );
        }
        ExampleTest extest = new ExampleTest();
        extest.testExample1( ex1 );
        extest.testExample2( ex2 );
        extest.testTest( test );
    }
}
