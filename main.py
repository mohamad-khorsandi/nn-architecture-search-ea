from nas import NAS


def main():
    nas = NAS(10, 10, 2, .9, .9, .4)
    nas.run()
    nas.show_statistics()


if __name__ == '__main__':
    main()


