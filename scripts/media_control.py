import os, vlc, glob, random

class MediaController:
    def __init__(self, music_folder=r"C:\Users\prems\Music"):
        # Create only ONE VLC instance
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        self.music_folder = music_folder
        self.playlist = glob.glob(os.path.join(music_folder, "*.mp3"))
        self.index = 0
        self.volume = 100
        self.player.audio_set_volume(self.volume)

    def play(self):
        if not self.playlist:
            print("[ERROR] No music files found!")
            return
        media = self.instance.media_new(self.playlist[self.index])
        self.player.set_media(media)
        self.player.play()
        print(f"[ACTION] Play: {self.playlist[self.index]}")

    def pause(self):
        if self.player.is_playing():
            self.player.pause()
            print("[ACTION] Pause")

    def next(self):
        if not self.playlist:
            return
        self.index = (self.index + 1) % len(self.playlist)
        media = self.instance.media_new(self.playlist[self.index])
        self.player.set_media(media)
        self.player.play()
        print(f"[ACTION] Next: {self.playlist[self.index]}")

    def volume_up(self):
        self.volume = min(200, self.volume + 10)
        self.player.audio_set_volume(self.volume)
        print(f"[ACTION] Volume Up: {self.volume}")

    def volume_down(self):
        self.volume = max(0, self.volume - 10)
        self.player.audio_set_volume(self.volume)
        print(f"[ACTION] Volume Down: {self.volume}")
