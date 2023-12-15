class tooSmall(Exception):

    def __init__(self, name, a=1, b=0):

        super().__init__()

        self.name = name
        self.b = a

        if self.name == 'ALL':
            print(f"{self.name} constants are now {a} instead of whatever they were")
        elif self.name == 'ANY':
            print(f"There's more than one wrong constant and I can't handle it")
        else:
            print(f"{self.name} is now {a} instead of {b}")
