package io.github.mandar2812.PlasmaML.cdf.record;

import io.github.mandar2812.PlasmaML.cdf.Buf;

import java.io.IOException;
import java.io.InputStream;

/**
 * Buf implementation based on an existing Buf instance.
 * All methods are delegated to the base buf.
 *
 * @author   Mark Taylor
 * @since    18 Jun 2013
 */
public class WrapperBuf implements Buf {

    private final Buf base_;

    /**
     * Constructor.
     *
     * @param  base  base buf
     */
    public WrapperBuf( Buf base ) {
        base_ = base;
    }

    public long getLength() {
        return base_.getLength();
    }

    public int readUnsignedByte( Pointer ptr ) throws IOException {
        return base_.readUnsignedByte( ptr );
    }

    public int readInt( Pointer ptr ) throws IOException {
        return base_.readInt( ptr );
    }

    public long readOffset( Pointer ptr ) throws IOException {
        return base_.readOffset( ptr );
    }

    public String readAsciiString( Pointer ptr, int nbyte ) throws IOException {
        return base_.readAsciiString( ptr, nbyte );
    }

    public void setBit64( boolean bit64 ) {
        base_.setBit64( bit64 );
    }

    public boolean isBit64() {
        return base_.isBit64();
    }

    public void setEncoding( boolean isBigendian ) {
        base_.setEncoding( isBigendian );
    }

    public boolean isBigendian() {
        return base_.isBigendian();
    }

    public void readDataBytes( long offset, int count, byte[] array )
            throws IOException {
        base_.readDataBytes( offset, count, array );
    }

    public void readDataShorts( long offset, int count, short[] array )
            throws IOException {
        base_.readDataShorts( offset, count, array );
    }

    public void readDataInts( long offset, int count, int[] array )
            throws IOException {
        base_.readDataInts( offset, count, array );
    }

    public void readDataLongs( long offset, int count, long[] array )
            throws IOException {
        base_.readDataLongs( offset, count, array );
    }

    public void readDataFloats( long offset, int count, float[] array )
            throws IOException {
        base_.readDataFloats( offset, count, array );
    }

    public void readDataDoubles( long offset, int count, double[] array )
            throws IOException {
        base_.readDataDoubles( offset, count, array );
    }

    public InputStream createInputStream( long offset ) {
        return base_.createInputStream( offset );
    }

    public Buf fillNewBuf( long count, InputStream in ) throws IOException {
        return base_.fillNewBuf( count, in );
    }
}
