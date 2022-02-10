
def get_DBLP(radio):
    ouput_filename_networkx = "data/DBLP/DBLP0/embedding/emb_s"+str(radio)+"0x.number2"
    ouput_filename_networky = "data/DBLP/DBLP1/embedding/emb_s"+str(radio)+"0x.number2"
    networkx_file = "data/DBLP/DBLP0/dblp_0_following.number"
    networky_file = "data/DBLP/DBLP1/dblp_1_following.number"
    anchor_file="data/DBLP/DBLP_0_1_groundtruth/dblp_0_1_groundtruth."+str(radio)+".foldtrain.train.number"
    test_file = "data/DBLP/DBLP_0_1_groundtruth/dblp_0_1_groundtruth."+str(radio)+".foldtrain.test.number"
    return ouput_filename_networkx,ouput_filename_networky,networkx_file,networky_file,anchor_file,test_file

def get_ACMDBLP(radio):
    ouput_filename_networkx = "data/ACM-DBLP/acm/embedding/emb_s" + str(radio) + "0x.number"
    ouput_filename_networky = "data/ACM-DBLP/dblp/embedding/emb_s" + str(radio) + "0x.number"
    networkx_file = "data/ACM-DBLP/acm/edges.follow.number"
    networky_file = "data/ACM-DBLP/dblp/edges.follow.number"
    anchor_file = "data/ACM-DBLP/anchor_set/train_" + str(radio) + "_number"
    test_file = "data/ACM-DBLP/anchor_set/test_" + str(radio) + "_number"
    return ouput_filename_networkx, ouput_filename_networky, networkx_file, networky_file, anchor_file, test_file

def get_FT(radio):
    ouput_filename_networkx = "data/Foursquare-Twitter/foursquare/embeddings/emb_s" + str(radio) + "0x.number2"
    ouput_filename_networky = "data/Foursquare-Twitter/twitter/embeddings/emb_s" + str(radio) + "0x.number2"
    networkx_file = "data/Foursquare-Twitter/foursquare/following.number"
    networky_file = "data/Foursquare-Twitter/twitter/following.number"
    anchor_file = "data/Foursquare-Twitter/twitter_foursquare_groundtruth/groundtruth." + str(radio) + ".foldtrain.train.number"
    test_file = "data/Foursquare-Twitter/twitter_foursquare_groundtruth/groundtruth." + str(radio) + ".foldtrain.test.number"
    return ouput_filename_networkx, ouput_filename_networky, networkx_file, networky_file, anchor_file, test_file

def get_phonemail(radio):
    ouput_filename_networkx = "data/phone-email/email/embedding/emb_s" + str(radio) + "0x.number2"
    ouput_filename_networky = "data/phone-email/phone/embedding/emb_s" + str(radio) + "0x.number2"
    networkx_file = "data/phone-email/email/email_following.number"
    networky_file = "data/phone-email/phone/phone_following.number"
    anchor_file = "data/phone-email/traintest/train_" + str(radio) + ".number"
    test_file = "data/phone-email/traintest/test_" + str(radio) + ".number"
    return ouput_filename_networkx, ouput_filename_networky, networkx_file, networky_file, anchor_file, test_file

def get_data(ratio,dataset):
    if dataset=='phone-email':
        return get_phonemail(ratio)
    if dataset=='Foursquare-Twitter':
        return get_FT(ratio)
    if dataset=='ACM-DBLP':
        return get_ACMDBLP(ratio)
    return get_ACMDBLP(ratio)