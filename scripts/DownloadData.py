import os
import wget

url_dict = {
    "train_pbmc": "https://www.dropbox.com/s/wk5zewf2g1oat69/train_pbmc.h5ad?dl=1",
    "valid_pbmc": "https://www.dropbox.com/s/nqi971n0tk4nbfj/valid_pbmc.h5ad?dl=1",

    "train_hpoly": "https://www.dropbox.com/s/7ngt0hv21hl2exn/train_hpoly.h5ad?dl=1",
    "valid_hpoly": "https://www.dropbox.com/s/bp6geyvoz77hpnz/valid_hpoly.h5ad?dl=1",

    "train_species": "https://www.dropbox.com/s/eprgwhd98c9quiq/train_species.h5ad?dl=1",
    "valid_species": "https://www.dropbox.com/s/bwq18z0mzy6h5d7/valid_species.h5ad?dl=1",

}

def download_data(data_name, key=None):
    path = os.getcwd()
    
    if not os.path.isdir(os.path.abspath(os.path.join(path, os.pardir))+"/data/"):
        os.makedirs(os.path.abspath(os.path.join(path, os.pardir))+"/data/")
        
    data_path = os.path.abspath(os.path.join(path, os.pardir))+"/data/"
    if key is None:
        train_path = os.path.join(data_path, f"{data_name}.h5ad")

        train_url = url_dict[f"train_{data_name}"]
        valid_url = url_dict[f"valid_{data_name}"]

        if not os.path.exists(train_path):
            wget.download(train_url, train_path)
    else:
        data_path = os.path.join(data_path, f"{key}.h5ad")
        data_url = url_dict[key]

        if not os.path.exists(data_path):
            wget.download(data_url, data_path)
    print(f"{data_name} data has been downloaded and saved in {data_path}")


def main():
    data_names = ["pbmc", "hpoly", "species"]
    for data_name in data_names:
        print(data_name)
        download_data(data_name)

if __name__ == "__main__":
    main()