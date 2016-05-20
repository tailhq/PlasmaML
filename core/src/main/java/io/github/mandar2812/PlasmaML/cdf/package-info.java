/**
 * Pure java library for read-only access to CDF (NASA Common Data Format)
 * files.
 *
 * <p>For low-level access to the record data of a CDF file, use the
 * {@link io.github.mandar2812.PlasmaML.cdf.CdfReader} class.
 * For high-level access to the variables and attributes that form
 * the CDF data and metadata, use the
 * {@link io.github.mandar2812.PlasmaML.cdf.CdfContent} class.
 *
 * <p>The package makes extensive use of NIO buffers for mapped read-on-demand
 * data access, so should be fairly efficient for reading scalar records
 * and whole raw array records.  Convenience methods for reading shaped
 * arrays may be less efficient.
 *
 * <p>This package is less capable than the official JNI-based
 * java interface to the CDF C library (read only, less flexible data read
 * capabilities), but it is pure java (no native code required) and it's
 * also quite a bit less complicated to use.
 */
package io.github.mandar2812.PlasmaML.cdf;
