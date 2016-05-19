package io.github.mandar2812.PlasmaML.cdf;

import io.github.mandar2812.PlasmaML.cdf.record.Record;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks field members of {@link Record} subclasses which represent
 * absolute file offsets.  Fields marked with this annotation must also
 * be marked with {@link CdfField}, and must be of type
 * <code>Long</code> or <code>long[]</code>.
 *
 * @author   Mark Taylor
 * @since    26 Jun 2013
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface OffsetField {
}
