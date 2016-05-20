package io.github.mandar2812.PlasmaML.cdf.record;

/**
 * Keeps track of a file offset.
 *
 * @author   Mark Taylor
 * @since    18 Jun 2013
 */
public class Pointer {

    private long value_;

    /**
     * Constructor.
     *
     * @param  value  initial value
     */
    public Pointer( long value ) {
        value_ = value;
    }

    /**
     * Returns this pointer's current value.
     *
     * @return  value
     */
    public long get() {
        return value_;
    }

    /**
     * Returns this pointer's current value and increments it by a given step.
     *
     * @param   increment  amount to increase value by
     * @return   pre-increment value
     */
    public long getAndIncrement( int increment ) {
        long v = value_;
        value_ += increment;
        return v;
    }
   
    /**
     * Sets this pointer's current value.
     *
     * @param   value  new value
     */
    public void set( long value ) {
        value_ = value;
    }
}
