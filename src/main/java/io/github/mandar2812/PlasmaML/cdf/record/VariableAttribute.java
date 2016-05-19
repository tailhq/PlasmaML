package io.github.mandar2812.PlasmaML.cdf.record;

import io.github.mandar2812.PlasmaML.cdf.AttributeEntry;

/**
 * Provides the description and per-variable entry values
 * for a CDF attribute with variable scope.
 *
 * @author   Mark Taylor
 * @since    20 Jun 2013
 */
public class VariableAttribute {

    private final String name_;
    private final AttributeEntry[] rEntries_;
    private final AttributeEntry[] zEntries_;

    /**
     * Constructor.
     *
     * @param  name  attribute name
     * @param  rEntries  rEntry values for this attribute
     * @param  zEntries  zEntry values for this attribute
     */
    public VariableAttribute( String name, AttributeEntry[] rEntries,
                              AttributeEntry[] zEntries ) {
        name_ = name;
        rEntries_ = rEntries;
        zEntries_ = zEntries;
    }

    /**
     * Returns this attribute's name.
     *
     * @return  attribute name
     */
    public String getName() {
        return name_;
    }

    /**
     * Returns the entry value that a given variable has for this attribute.
     * If the variable has no entry for this attribute, null is returned.
     *
     * @param  variable  CDF variable from the same CDF as this attribute
     * @return   this attribute's value for <code>variable</code>
     */
    public AttributeEntry getEntry( Variable variable ) {
        AttributeEntry[] entries = variable.isZVariable() ? zEntries_
                                                          : rEntries_;
        int ix = variable.getNum();
        return ix < entries.length ? entries[ ix ] : null;
    }
}
