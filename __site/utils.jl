function hfun_bar(vname)
  val = Meta.parse(vname[1])
  return round(sqrt(val), digits=2)
end

function hfun_m1fill(vname)
  var = vname[1]
  return pagevar("index", var)
end

function lx_baz(com, _)
  # keep this first line
  brace_content = Franklin.content(com.braces[1]) # input string
  # do whatever you want here
  return uppercase(brace_content)
end

# Thanks @tlienart

using Franklin, JSON

using Markdown, Dates

include("youtube_videos.jl")

const DATEFMT = dateformat"yyyy-mm-dd HH:MMp"
const TZ = "America/New_York"

function hfun_doc(params)
    fname = join(params[1:max(1, length(params)-2)], " ")
    head = params[end-1]
    type = params[end]
    doc = eval(Meta.parse("@doc $fname"))
    txt = Markdown.plain(doc)
    # possibly further processing here
    body = Franklin.fd2html(txt, internal=true)
    return """
      <div class="docstring">
          <h2 class="doc-header" id="$head">
            <a href="#$head">$head</a>
            <div class="doc-type">$type</div></h2>
          <div class="doc-content">$body</div>
      </div>
    """
end

function hfun_youtube_placeholder(params)
    id = params[1]
    return """
    <div id="videoContainer" >
            <div id="player"></div>        
    </div>
    <script>
            var tag = document.createElement('script');

            tag.src = "https://www.youtube.com/iframe_api";
            var firstScriptTag = document.getElementsByTagName('script')[0];
            firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

            var player;
            function onYouTubeIframeAPIReady() {
              player = new YT.Player('player', {
                height: '300',
                width: '100%',
                videoId: '$(get(videos, id, id))',
                playerVars: { 'autoplay': 0, 'rel': 0, 'cc_load_policy': 1  }
              });
            }
    </script>
<script>
function changeYouTubeSource(startTime, endTime) {
 
  // Get the src attribute of the YouTube iframe.
  var youtubeIframe = document.getElementById('player');
 
  var youtubeIframeSrc = document.getElementById('player').getAttribute('src');
 
  // Variables for assigning in the condition below.
  var trimmedIframeUrl = '';
  var iframeUrlTimeStamp = '';
 
  // If the src attribute URL already contains a media fragment, remove it.
  if (youtubeIframeSrc.match(/&start=/g) ) {
      var mediaFragmentIndex = youtubeIframeSrc.indexOf('&start=');
      trimmedIframeUrl = youtubeIframeSrc.slice(0, mediaFragmentIndex);
 
      if (endTime === 0) {
          iframeUrlTimeStamp = trimmedIframeUrl + '&start=' + startTime;
      } else {
          iframeUrlTimeStamp = trimmedIframeUrl + '&start=' + startTime + '&end=' + endTime;
      }
  }
 
  // If the src attribute URL doesn't contain a media fragment, add one.
  if (youtubeIframeSrc.match(/&start=/g) === null) {
      if (endTime === 0) {
          iframeUrlTimeStamp = youtubeIframeSrc + '&start=' + startTime;
      } else {
          iframeUrlTimeStamp = youtubeIframeSrc + '&start=' + startTime + '&end=' + endTime;
      }
  }
  // 1 second delay to allow for scrolling to video.
    setTimeout(function() {
        // Enable autoplay on the new URL.
        var iframeAutoplayUrl = iframeUrlTimeStamp.replace('autoplay=0', 'autoplay=1' );
 
        // Set the src attribute as the original URL with the media fragment appended.
        youtubeIframe.setAttribute('src', iframeAutoplayUrl);
    }, 1000);
}
</script>
    """
end

function hfun_yt_tsp(params)
    start = params[1]
    final = params[2]
    s = Time(0) + Second(start)
    if second(s) < 10
      display = "$(minute(s)):0$(second(s))"
    elseif hour(s) < 1
      display = "$(minute(s)):$(second(s))"
    else
      display = display = "$(hour(s)):0$(minute(s)):$(second(s))"
    end
    title = join(params[3:end], " ")
    return """
    <br>
    <a href='#player' onclick='changeYouTubeSource($start,$final)'> $display</a> $title
    """ 
end

function hfun_youtube(params)
    id = params[1]
    return """
    <iframe id="$id" width="100%" height="300"
    src="https://www.youtube.com/embed/$(get(videos, id, id))"
    frameborder="0"
    allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
    """
end

function hfun_youtube_start(params)
  id = params[1]
  start = params[2]
  return """
  <iframe id="$id" width="50%" height="30"
  src="https://www.youtube.com/embed/$(get(videos, id, id))?start=$start"
  frameborder="0"
  allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
  allowfullscreen></iframe>
  """
end

function hfun_showtime(params)
    id = params[1]
    str = locvar(id)
    if isnothing(str)
        @warn "Unknown datetime variable $str"
        return ""
    end
    try
        DateTime(str, DATEFMT)
    catch err
        @warn "There was an error parsing date $str, the format is yyyy-mm-dd HH:MMp (see ?DateFormat)"
        rethrow(err)
    end
end


function parse_duration(str)
    str = replace(str, r"^PT"=>"")
    hrex, mrex, srex = Regex.(string.("^([0-9]+)", ["H","M","S"]))

    t = 0
    hmatch = match(hrex, str)
    if !isnothing(hmatch)
        h = parse(Int, hmatch[1])
        t += 60*60*h
        str = replace(str, hrex=>"")
    end

    mmatch = match(mrex, str)
    if !isnothing(mmatch)
        m = parse(Int, mmatch[1])
        t += 60*m
        str = replace(str, mrex=>"")
    end

    smatch = match(srex, str)
    if !isnothing(smatch)
        s = parse(Int, smatch[1])
        t += s
        str = replace(str, srex=>"")
    end

    t
end

function hfun_go_live()
    seq = locvar("sequence")
    airtime = locvar("airtime")

    if isnothing(seq)
        @warn "airtime set, but no `sequence` variable not defined." *
        "sequence is an array of video IDs to play in order on this page"
    end

    vid_ids = [get(videos, s, s) for s in seq]

    f = tempname()
    # Get the duration of each video
    download("https://www.googleapis.com/youtube/v3/videos?id=$(join(vid_ids, ","))&part=contentDetails&key=AIzaSyDZhbWHc2PTEFTx173MaTgddnWCGPqdbB8", f)
    dict = JSON.parse(String(read(f)))

    durations = [parse_duration(video["contentDetails"]["duration"])
                 for video in dict["items"]]


    jrepr(x) = sprint(io->JSON.print(io, x))
    """
    <script src="/assets/moment.min.js"></script>
    <script src="/assets/moment-timezone.js"></script>
    <script src="/assets/live-player.js"></script>
    <script>
    play_live($(jrepr(string(DateTime(airtime, DATEFMT)))), $(jrepr(TZ)), $(jrepr(seq)), $(jrepr(durations)))
    </script>
    """
end
