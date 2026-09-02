from pathlib import Path
from movie_broll.inspect_source import normalize_ffprobe
def test_normalizes_optional_fields_and_tracks(tmp_path):
 movie=tmp_path/"movie.mp4"; movie.write_bytes(b"x")
 payload={"format":{"duration":"10.0","format_name":"mov,mp4"},"streams":[{"index":0,"codec_type":"video","codec_name":"h264","width":1920,"height":1080,"avg_frame_rate":"24000/1001","r_frame_rate":"24/1","time_base":"1/24000"},{"index":1,"codec_type":"audio","codec_name":"aac","sample_rate":"48000","channels":2},{"index":2,"codec_type":"audio","codec_name":"aac","tags":{"language":"spa"}}]}
 data=normalize_ffprobe(payload,movie)
 assert data["video"]["bitrate"] is None and data["video"]["nb_frames"] is None
 assert len(data["audio_tracks"])==2 and data["subtitle_streams"]["count"]==0
