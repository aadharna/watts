import yaml


class GameSchema:
    def __init__(self, gdy_file):
        self._unpacked_game = self._unpack_gdy(gdy_file)

        self.agent_chars = self._get_agent_chars_from_gdy()
        self.wall_char = self._get_wall_char_from_gdy()
        self.interesting_chars = self._get_other_chars_from_gdy()
        self.n_human_levels = len(self._unpacked_game['Environment']['Levels'])
        self.index_to_str, self.str_to_index = self._make_int_char_map()

        assert(self.agent_chars is not None)
        assert(len(self.interesting_chars) != 0)

    @staticmethod
    def _unpack_gdy(gdy_file):
        with open(gdy_file, 'r') as file:
            return yaml.safe_load(file)

    def _get_agent_chars_from_gdy(self):
        for obj in self._unpacked_game['Objects']:
            if obj['Name'] == self._unpacked_game['Environment']['Player']['AvatarObject']:
                if 'Count' in self._unpacked_game['Environment']['Player']:
                    count = self._unpacked_game['Environment']['Player']['Count']
                    return {f'{obj["MapCharacter"]}{i}' for i in range(1, count + 1)}
                else:
                    return {obj['MapCharacter']}

    def _get_wall_char_from_gdy(self):
        # Assume wall character is along the edges, so get wall char from top left corner
        return self._unpacked_game['Environment']['Levels'][0][0][0]

    def _get_other_chars_from_gdy(self):
        other_chars = set()
        for obj in self._unpacked_game['Objects']:
            this_char = obj['MapCharacter']
            # Get the first character of one the agent chars to properly handle multiagent
            agent_char = ''
            for a_char in self.agent_chars:
                agent_char = a_char[0]
                break
            if this_char != agent_char and this_char != self.wall_char:
                other_chars.add(this_char)
        return other_chars

    def _make_int_char_map(self):
        index_to_str = {}
        str_to_index = {}
        for i, ob_dict in enumerate(self._unpacked_game['Objects']):
            index_to_str[i] = ob_dict['MapCharacter']
            str_to_index[ob_dict['MapCharacter']] = i

        index_to_str[i+1] = '.'
        str_to_index['.'] = i+1
        return index_to_str, str_to_index

if __name__ == '__main__':
    import os

    schema = GameSchema(os.path.join('..', 'example_levels', 'limited_zelda.yaml'))
    print(schema.str_to_index)
    print(schema.index_to_str)
    print(schema.n_human_levels)