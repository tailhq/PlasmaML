package io.github.mandar2812.PlasmaML.cdf;

import java.io.IOException;
import java.io.InputStream;

/**
 * Decompression stream for CDF's version of Run Length Encoding.
 *
 * <p>The compressed stream is just like the uncompressed one,
 * except that a byte with the special value V is followed by
 * a byte giving the number of additional bytes V to consider present
 * in the stream.
 * Thus the compressed stream:
 * <blockquote>
 *    1 2 3 0 0 4 5 6 0 2
 * </blockquote>
 * is decompressed as 
 * <blockquote>
 *    1 2 3 0 4 5 6 0 0 0 
 * </blockquote>
 * (assuming a special value V=0).
 *
 * <p>This format was deduced from reading the cdfrle.c source file
 * from the CDF distribution.
 * 
 * @author   Mark Taylor
 * @since    17 May 2013
 */
public class RunLengthInputStream extends InputStream {

    private final InputStream base_;
    private final int rleVal_;
    private int vCount_;

    /**
     * Constructor.
     *
     * @param  base   input stream containing RLE-compressed data
     * @param  rleVal  the byte value whose run lengths are compressed
     *                 (always zero for CDF as far as I can tell)
     */
    public RunLengthInputStream( InputStream base, byte rleVal ) {
        base_ = base;
        rleVal_ = rleVal & 0xff;
    } 

    @Override
    public int read() throws IOException {
        if ( vCount_ > 0 ) {
            vCount_--;
            return rleVal_;
        }
        else {
            int b = base_.read();
            if ( b == rleVal_ ) {
                int c = base_.read();
                if ( c >= 0 ) {
                    vCount_ = c;
                    return rleVal_;
                }
                else {
                    throw new CdfFormatException( "Bad RLE data" );
                }
            }
            else {
                return b;
            }
        }
    }

    @Override
    public int available() throws IOException {
        return base_.available() + vCount_;
    }

    @Override
    public void close() throws IOException {
        base_.close();
    }

    @Override
    public boolean markSupported() {
        return false;
    }
}
