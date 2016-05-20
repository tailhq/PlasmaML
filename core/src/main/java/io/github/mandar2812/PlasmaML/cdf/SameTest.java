package io.github.mandar2812.PlasmaML.cdf;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

import io.github.mandar2812.PlasmaML.cdf.record.VariableAttribute;
import io.github.mandar2812.PlasmaML.cdf.record.Variable;

/**
 * Tests that multiple specified CDF files identical CDF content.
 * The second, third, fourth, ... -named files are compared with the
 * first-named one.
 * Any discrepancies are reported with context.
 * The error count can be obtained.
 *
 * @author   Mark Taylor
 * @since    25 Jun 2013
 */
public class SameTest {

    private final File[] files_;
    private final PrintStream out_;
    private int nerror_;
    private Stack<String> context_;

    /**
     * Constructor.
     *
     * @param   files  nominally similar files to assess
     */
    public SameTest( File[] files, PrintStream out ) {
        files_ = files;
        out_ = out;
        context_ = new Stack<String>();
    }

    /**
     * Runs the comparisons.
     */
    public void run() throws IOException {
        CdfContent c0 = new CdfContent( new CdfReader( files_[ 0 ] ) );
        context_.clear();
        for ( int i = 1; i < files_.length; i++ ) {
            pushContext( files_[ 0 ].getName(), files_[ i ].getName() );
            compareCdf( c0, new CdfContent( new CdfReader( files_[ i ] ) ) );
            popContext();
        }
        if ( nerror_ > 0 ) {
            out_.println( "Error count: " + nerror_ );
        }
    }

    /**
     * Returns the number of errors found.
     */
    public int getErrorCount() {
        return nerror_;
    }

    /**
     * Compares two CDFs for equivalence.
     */
    private void compareCdf( CdfContent cdf0, CdfContent cdf1 )
            throws IOException {
        pushContext( "Global Attributes" );
        List<Pair<GlobalAttribute>> gattPairs =
            getPairs( cdf0.getGlobalAttributes(),
                      cdf1.getGlobalAttributes() );
        popContext();
        pushContext( "Variable Attributes" );
        List<Pair<VariableAttribute>> vattPairs =
            getPairs( cdf0.getVariableAttributes(),
                      cdf1.getVariableAttributes() );
        popContext();
        pushContext( "Variables" );
        List<Pair<Variable>> varPairs =
            getPairs( cdf0.getVariables(), cdf1.getVariables() );
        popContext();

        pushContext( "Global Attributes" );
        for ( Pair<GlobalAttribute> gattPair : gattPairs ) {
            compareGlobalAttribute( gattPair.item0_, gattPair.item1_ );
        }
        popContext();

        pushContext( "Variable Attributes" );
        for ( Pair<VariableAttribute> vattPair : vattPairs ) {
            compareVariableAttribute( vattPair.item0_, vattPair.item1_,
                                      varPairs );
        }
        popContext();

        pushContext( "Variables" );
        for ( Pair<Variable> varPair : varPairs ) {
            compareVariable( varPair.item0_, varPair.item1_ );
        }
        popContext();
    }

    /**
     * Compares two global attributes for equivalence.
     */
    private void compareGlobalAttribute( GlobalAttribute gatt0,
                                         GlobalAttribute gatt1 ) {
        pushContext( gatt0.getName(), gatt1.getName() );
        compareScalar( gatt0.getName(), gatt1.getName() );
        List<Pair<AttributeEntry>> entryPairs =
            getPairs( gatt0.getEntries(), gatt1.getEntries() );
        for ( Pair<AttributeEntry> entryPair : entryPairs ) {
            compareEntry( entryPair.item0_, entryPair.item1_ );
        }
        popContext();
    }

    /**
     * Compares two variable attributes for equivalence.
     */
    private void compareVariableAttribute( VariableAttribute vatt0,
                                           VariableAttribute vatt1,
                                           List<Pair<Variable>> varPairs ) {
        pushContext( vatt0.getName(), vatt1.getName() );
        compareScalar( vatt0.getName(), vatt1.getName() );
        for ( Pair<Variable> varPair : varPairs ) {
            pushContext( varPair.item0_.getName(), varPair.item1_.getName() );
            compareEntry( vatt0.getEntry( varPair.item0_ ),
                          vatt1.getEntry( varPair.item1_ ) );
            popContext();
        }
        popContext();
    }

    /**
     * Compares two variables for equivalence.
     */
    private void compareVariable( Variable var0, Variable var1 )
            throws IOException {
        pushContext( var0.getName(), var1.getName() );
        compareInt( var0.getNum(), var1.getNum() );
        compareScalar( var0.getName(), var1.getName() );
        compareScalar( var0.getDataType(), var1.getDataType() );
        Object work0 = var0.createRawValueArray();
        Object work1 = var1.createRawValueArray();
        int nrec = Math.max( var0.getRecordCount(), var1.getRecordCount() );
        for ( int irec = 0; irec < nrec; irec++ ) {
            pushContext( "rec#" + irec );
            compareValue( var0.readShapedRecord( irec, false, work0 ),
                          var1.readShapedRecord( irec, false, work1 ) );
            compareValue( var0.readShapedRecord( irec, true, work0 ),
                          var1.readShapedRecord( irec, true, work1 ) );
            popContext();
        }
        if ( nrec > 0 ) {
            // see readShapedRecord contract.
            assert var0.readShapedRecord( 0, false, work0 ) != work0;
            assert var1.readShapedRecord( 1, false, work1 ) != work1;
            assert var0.readShapedRecord( 0, true, work0 ) != work0;
            assert var1.readShapedRecord( 1, true, work1 ) != work1;
        }
        popContext();
    }

    /**
     * Compares two integers for equivalence.
     */
    private void compareInt( int i0, int i1 ) {
        compareScalar( new Integer( i0 ), new Integer( i1 ) );
    }

    /**
     * Compares two attribute entries for equivalence.
     */
    private void compareEntry( AttributeEntry ent0, AttributeEntry ent1 ) {
        boolean nul0 = ent0 == null;
        boolean nul1 = ent1 == null;
        if ( nul0 && nul1 ) {
            return;
        }
        else if ( nul0 || nul1 ) {
            error( "Entry nullness mismatch" );
        }
        else {
            compareScalar( ent0.getDataType(), ent1.getDataType() );
            compareScalar( ent0.getItemCount(), ent1.getItemCount() );
            Object va0 = ent0.getRawValue();
            Object va1 = ent1.getRawValue();
            for ( int i = 0; i < ent0.getItemCount(); i++ ) {
                pushContext( "#" + i );
                compareValue( ent0.getDataType().getScalar( va0, i ),
                              ent1.getDataType().getScalar( va1, i ) );
                popContext();
            }
        }
    }

    /**
     * Compares two scalar objects for equivalence.
     */
    private void compareScalar( Object v0, Object v1 ) {
        boolean match = v0 == null ? v1 == null : v0.equals( v1 );
        if ( ! match ) {
            error( "Value mismatch: " + quote( v0 ) + " != " + quote( v1 ) );
        }
    }

    /**
     * Compares to array values for equivalence.
     */
    private void compareArray( Object arr0, Object arr1 ) {
        int narr0 = Array.getLength( arr0 );
        int narr1 = Array.getLength( arr1 );
        if ( narr0 != narr1 ) {
            error( "Length mismatch: " + narr0 + " != " + narr1 );
        }
        int count = Math.min( narr0, narr1 );
        for ( int i = 0; i < count; i++ ) {
            pushContext( "el#" + i );
            compareScalar( Array.get( arr0, i ), Array.get( arr1, i ) );
            popContext();
        }
    }

    /**
     * Compares two miscellaneous objects for equivalence.
     */
    private void compareValue( Object v0, Object v1 ) {
        Object vt = v0 == null ? v1 : v0;
        if ( vt == null ) {
        }
        else if ( vt.getClass().getComponentType() != null ) {
            compareArray( v0, v1 );
        }
        else {
            compareScalar( v0, v1 );
        }
    }

    /**
     * Quotes an object string representation for output.
     */
    private String quote( Object obj ) {
        return obj instanceof String ? ( "\"" + obj + "\"" )
                                     : String.valueOf( obj );
    }

    /**
     * Pushes a context frame labelled by two, possibly identical, strings.
     */
    private void pushContext( String label0, String label1 ) {
        pushContext( label0.equals( label1 ) ? label0
                                             : ( label0 + "/" + label1 ) );
    }

    /**
     * Pushes a labelled context frame.
     */
    private void pushContext( String label ) {
        context_.push( label );
    }

    /**
     * Pops a context frame from the stack.
     */
    private void popContext() {
        context_.pop();
    }

    /**
     * Emits an message about equivalence failure with context.
     */
    private void error( String msg ) {
        out_.println( context_.toString() + ": " + msg );
        nerror_++;
    }
                                            
    /**
     * Turns a pair of presumed corresponding arrays into a list of pairs.
     */
    private <T> List<Pair<T>> getPairs( T[] arr0, T[] arr1 ) {
        if ( arr1.length != arr0.length ) {
            error( "Array length mismatch: "
                 + arr0.length + " != " + arr1.length );
        }
        int count = Math.min( arr0.length, arr1.length );
        List<Pair<T>> list = new ArrayList<Pair<T>>( count );
        for ( int i = 0; i < count; i++ ) {
            list.add( new Pair<T>( arr0[ i ], arr1[ i ] ) );
        }
        return list;
    }

    /**
     * Groups two objects.
     */
    private static class Pair<T> {
        final T item0_;
        final T item1_;
        Pair( T item0, T item1 ) {
            item0_ = item0;
            item1_ = item1;
        }
    }

    /**
     * Main method.  Supply filenames of any number of
     * nominally similar CDF files as arguments.
     * The run will exit with a non-zero status if any discrepancies are found.
     */
    public static void main( String[] args ) throws IOException {
        File[] files = new File[ args.length ];
        for ( int i = 0; i < args.length; i++ ) {
            files[ i ] = new File( args[ i ] );
        }
        SameTest test = new SameTest( files, System.err );
        test.run();
        if ( test.getErrorCount() > 0 ) {
            System.exit( 1 );
        }
    }
}
