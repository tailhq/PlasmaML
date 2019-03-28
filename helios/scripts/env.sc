import ammonite.ops._

val summary_dirs = Map(
    "citronelle" -> root/'home/System.getProperty("user.name")/'tmp,
    "juniper"    -> root/'export/'scratch3/System.getProperty("user.name")/'summaries,
    "wax"        -> root/'export/'scratch2/System.getProperty("user.name")/'summaries
)

val data_dirs = Map(
    "citronelle" -> root/'home/System.getProperty("user.name")/"data_repo"/'helios,
    "juniper"    -> root/'export/'scratch2/System.getProperty("user.name")/"data_repo"/'helios
)

val host: Option[String] = try {
    Some(java.net.InetAddress.getLocalHost().toString.split('/').head.split('.').head)
} catch {
    case _: java.net.UnknownHostException => None
    case _: Exception => None
}

val default_summary_dir = home/'tmp
val default_data_dir    = pwd/'data

val summary_dir = host match {
    case None => default_summary_dir
    case Some(host) => if(summary_dirs.contains(host)) summary_dirs(host) else default_summary_dir
}

val data_dir = host match {
    case None => default_data_dir
    case Some(host) => if(data_dirs.contains(host)) data_dirs(host) else default_data_dir
}
