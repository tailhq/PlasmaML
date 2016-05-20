package io.github.mandar2812.PlasmaML.cdf;

import io.github.mandar2812.PlasmaML.cdf.record.Record;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks field members of {@link Record} subclasses which correspond directly
 * to fields in typed CDF records in a CDF file.
 *
 * <p>These fields are all public and final, and have names matching
 * (apart perhaps from minor case tweaking)
 * the fields documented in the relevant subsections of Section 2 of the
 * CDF Internal Format Description document.
 * 
 * <p>See that document for a description of the meaning of these fields.
 *
 * @author   Mark Taylor
 * @since    25 Jun 2013
 * @see
 * <a href="http://cdaweb.gsfc.nasa.gov/pub/software/cdf/doc/cdf34/cdf34ifd.pdf"
 *    >CDF Internal Format Description document</a>
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface CdfField {
} 
