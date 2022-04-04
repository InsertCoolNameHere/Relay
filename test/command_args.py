import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Individual ML Trainer Module')
    parser.add_argument('-sp', '--save_path', help='Save Path for Trained Parents', required=False,
                        default="/saved_models/")
    parser.add_argument('-hs', '--host_server', help='Coordinator Machine Name', required=False,
                        default="kiwis.cs.colostate.edu")
    parser.add_argument('-hp', '--host_port', help='Port for The Coordinator Machine', required=False, type=int,
                        default=31477)
    args = vars(parser.parse_args())

    model_save_path = args['save_path']
    host = args['host_server']
    port = args['host_port']

    print(type(port), (port+1))
