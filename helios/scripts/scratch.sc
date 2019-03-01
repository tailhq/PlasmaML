import ammonite.ops._
import ammonite.ops.ImplicitWd._

val script1 = pwd/'helios/'scripts/"visualise_tl_results.R"
val script2 = pwd/'helios/'scripts/"visualise_tl_preds.R"

val search_dir = ls! home/'Downloads |? (_.segments.last.contains("results_exp"))

val relevant_dirs = search_dir || ((p: Path) => ls! p |? (_.isDir)) // | (_.head) 

relevant_dirs.foreach(dir => {
    
    print("Processing directory ")
    pprint.pprintln(dir)
    
    val pre = 
        if(dir.segments.last.contains("const_v")) "exp2_" 
        else if(dir.segments.last.contains("const_a")) "exp3_" 
        else if (dir.segments.last.contains("softplus")) "exp4_" 
        else ""

    try {
        %%('Rscript, script1, dir, pre)
    } catch {
        case e: Exception => e.printStackTrace()
    }
    
    try {
        %%('Rscript, script2, dir, pre)
    } catch {
        case e: Exception => e.printStackTrace()
    }
    
})


