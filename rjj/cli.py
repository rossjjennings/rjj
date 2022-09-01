class ProgressBar:
    """
    Display a progress bar during an ongoing computation.
    """
    def __init__(self, width, style='block2'):
        self.width = width
        self.style = style
        self.status = self.get_string(0)

    def get_string(self, progress):
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be between 0 and 1 inclusive")

        if self.style.startswith('block') or self.style in ['dots', 'fade']:
            if self.style == 'dots':
                n = 8
                blocks = '⡀⡄⡆⡇⣇⣧⣷⣿'
                fg_char = '⣿'
            elif self.style == 'fade':
                n = 4
                blocks = ' ░▒▓█'
                fg_char = '█'
            else:
                n = int(self.style[5:])
                fg_char = '█'
                if n == 1:
                    blocks = '█'
                elif n == 2:
                    blocks = ' ▌█'
                elif n == 4:
                    blocks = ' ▎▌▊█'
                elif n == 8:
                    blocks = ' ▏▎▍▌▋▊▉█'
                else:
                    raise ValueError(f"style {self.style} not understood")
            
            if progress == 1.0:
                return '|' +fg_char*(self.width - 2) + '|'
            else:
                i = int(progress*(self.width - 2)*n)
                return '|' + fg_char*(i//n) + blocks[i % n] + ' '*(self.width-i//n-3) + '|'

        elif self.style in ['=>', '->', '=>.', '->.']:
            bg_char = '.' if self.style.endswith('.') else ' '
            fg_char = '=' if self.style.startswith('=') else '-'
            if progress == 0.0:
                return '[' + bg_char*(self.width - 2) + ']'
            elif progress == 1.0:
                return '[' + fg_char*(self.width - 2) + ']'
            else:
                i = int(progress*(self.width - 2))
                return '[' + fg_char*i + '>' + bg_char*(self.width - i - 3) + ']'

        else:
            raise ValueError(f"style {self.style} not understood")

    def update(self, progress):
        print('\r', end='')
        self.status = self.get_string(progress)
        if len(self.status) != self.width:
            print(f'Status length {len(self.status)} did not match width {self.width}!')
        print(self.status, end='', flush=True)
