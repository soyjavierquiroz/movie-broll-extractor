import pytest
from movie_broll.srt import cue_statistics, parse_srt_text, timestamp_to_seconds, validate_timeline
def test_normal_multiline_and_crlf():
 r=parse_srt_text("1\r\n00:00:01,250 --> 00:00:03,500\r\nHola\r\nmundo\r\n\r\n2\r\n00:00:05,000 --> 00:00:06,000\r\nAdiós\r\n")
 assert r.cues[0].text=="Hola mundo" and r.cues[0].as_dict()["duration_seconds"]==2.25
 assert r.cues[0].cue_id=="SRT_000001"
def test_timestamp_and_malformed():
 assert timestamp_to_seconds("01:02:03,004")==3723.004
 with pytest.raises(ValueError): timestamp_to_seconds("00:99:00,000")
 assert parse_srt_text("1\nnot-a-time\ntext").malformed
def test_statistics_gaps_and_ordering():
 cues=parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\na\n\n2\n00:00:05,000 --> 00:00:07,000\nb").cues
 stats=cue_statistics(cues); assert stats["average_cue_duration"]==1.5 and stats["maximum_gap_seconds"]==3.0
 reversed_cues=list(reversed(cues)); assert validate_timeline(reversed_cues,10)["status"]=="ERROR"
