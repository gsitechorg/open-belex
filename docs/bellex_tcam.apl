
APL_C_INCLUDE <gsi/preproc_defs.h>
APL_C_INCLUDE <gsi/libapl.h>
APL_C_INCLUDE <gsi/libsys.h>
APL_C_INCLUDE <gsi/libgvml_element_wise.h>
APL_C_INCLUDE <gsi/gal-fast-funcs.h>

#include "gvml_apl_defs.apl.h"
/* smaps registers to hold predefined section maps.
 * These smaps registers shouldn't change during program execution.
 */
#define SM_0XFFFF SM_REG_4
#define SM_0X0001 SM_REG_5
// VRs used in TCAM:
#define DATA_VR 0
#define DATA_BAR_VR 1
#define M4 16				// VR for Plat index
#define UNARY_PRIORITY 17
#define TEMP 22
// Markers used in TCAM
#define MATCH_MRK 1<<4      // GVML_MRK0
#define VALID_MRK 1<<5      // GVML_MRK1

APL_FRAG hb_max_priority_search_rsp_read(RN_REG match_vr, SM_REG match_sm, RN_REG rsp_out_vr, RN_REG pri_unary_vr) // RSP2k = { priority [15:12], valid[11], index[ 10:0]}
{ // store best match from each half bank to RSP FIFO. If no matches found-
  // result is 0x0000.
	{ // AND-up the match_sm sections of match_vr
		match_sm: RL = SB[match_vr];
		match_sm: GL = RL;
	}
	SM_0XFFFF: RL = SB[pri_unary_vr] & GL;
  // OR-up the plats of RL, all the way up to one 16-bit vector for the half-bank
	SM_0XFFFF: RSP16 = RL;
	RSP256 = RSP16;
	RSP2K = RSP256;
	RSP_START_RET;
  // and back down
	RSP256 = RSP2K;
	RSP16 = RSP256;
	{	// get nibble_masks from RSP16
		SM_0XFFFF: RL =  SB[pri_unary_vr] ^ INV_RSP16;
		SM_0XFFFF: GL = RL;					// column with GL=1 -highest priority match ; If no mathc in hakf bank- GL =1 in all columns
	}
	{
		match_sm: RL = SB[match_vr] & GL;
		RSP_END;
	}
	// make sure if there is no match in whole bank- outputs is 0

	match_sm: GL = RL;
	// put result on RSP-FIFO
	SM_0XFFFF: RL = SB[rsp_out_vr] & GL;
	SM_0XFFFF: RSP16 = RL;
	RSP256 = RSP16;
	RSP2K = RSP256;
	RSP32K = RSP2K; //write to RSP FIFO
	//NOOP;
	NOOP;
	NOOP;
	RSP_END;
};

APL_FRAG search_TCAM(RN_REG valid_vr, SM_REG valid_sm, SM_REG query, RN_REG data, RN_REG data_bar, RN_REG match_store, SM_REG match_sm )
{ /*
    Perform a TCAM search over a 16-bit query stored in @match_sm.
    The data and data_bar hold the ternary value of the data and encoded as follows:
    Bit             ternary encodng
    '1' -           10
    '0' -           01
    'don't-care -   11
    reserved -      00
  */
	// set match bit = valid bit
	{
		valid_sm: RL = SB[valid_vr];
		valid_sm: GL = RL;						            // GL = Valid vit 0
	}
	// perform searches
	{
		query:  RL = SB[data] & GL;							// RL = TCAM_operation & valid
		~query: RL = SB[data_bar] & GL;
	}
	// store
	SM_0XFFFF:  GL = RL;
	match_sm:  SB[match_store]= GL;
};

uint16_t _bellex_single_TCAM_search(uint8_t* query){
	/*
	 Search @query in core, reutrn the best match Plat ID
	*/
		// if (gal_fast_apuc_id() !=0) return -1; // uncomment to use only APUC 0
	// initialize Rns and Sms
	apl_set_sm_reg(SM_REG_1, VALID_MRK);
	apl_set_sm_reg(SM_REG_2, MATCH_MRK);
	apl_set_rn_reg(RN_REG_0, DATA_VR);
	apl_set_rn_reg(RN_REG_1, DATA_BAR_VR);

    // Search first iteration from valid bit and store in match bit
	apl_set_sm_reg(SM_REG_0, query[0] | (query[1] << 8) );
	RUN_FRAG_ASYNC(search_TCAM(valid_vr = RN_REG_FLAGS, valid_sm = SM_REG_1,  query=SM_REG_0, data= RN_REG_0, data_bar=RN_REG_1, match_store= RN_REG_FLAGS, match_sm=SM_REG_2  ) );						// run slice 2 search

	for(uint iter = 1 ; iter <8 ; iter++){
		// search next iterations from Match bit and store in match bit
		apl_set_rn_reg(RN_REG_0, 2* iter + DATA_VR);
		apl_set_rn_reg(RN_REG_1, 2* iter + DATA_BAR_VR);
		apl_set_sm_reg(SM_REG_0, query[2*iter] | (query[2*iter+1] << 8) );
		RUN_FRAG_ASYNC(search_TCAM(valid_vr = RN_REG_FLAGS, valid_sm = SM_REG_2,  query=SM_REG_0, data= RN_REG_0, data_bar=RN_REG_1, match_store= RN_REG_FLAGS, match_sm=SM_REG_2  ) );						// run slice 2 search
	}
	// Perform half-bank best match
	apl_set_rn_reg(RN_REG_2, M4);
	apl_set_rn_reg(RN_REG_3, UNARY_PRIORITY);
	RUN_FRAG_ASYNC(hb_max_priority_search_rsp_read(match_vr =RN_REG_FLAGS, match_sm = SM_REG_2, rsp_out_vr =RN_REG_2, pri_unary_vr = RN_REG_3 ));
	// perform cross-bank best match (in this case: take match with highest index)
	uint16_t candidate[2];
	uint16_t best_match = 0 ;
	for(uint apc_id = 0 ; apc_id < 2; apc_id++)
	{
		apl_rsp_rd(apc_id);
		for (int bank = 0; bank < GSI_MMB_NUM_BANKS; bank++) {

				uint16_t rsp2k = apl_rd_rsp2k_reg(bank);
				// rsp2k += apc_id+ bank ;

				candidate[0] = (uint16_t) rsp2k; 			//lower bank match in the form of: bits [10:0] :column #. bit [11]: valid bit. bits [15:12] priority
				candidate[1] = (uint16_t) (rsp2k >>16); 		// Higher bank match
				if (candidate[0] > best_match){ //compare current best match and candidate result. Will update if candidate is valid AND has higher priority ( for same priority: see note above)
					best_match = candidate[0] & 0x0FFF;
				}
				if (candidate[1] > best_match)
				{
					best_match = candidate[1] & 0x0FFF;
				}
			}
	}
	return best_match;
}

int bellex_load_key_to_index(int index, uint8_t* rule, uint8_t priority)
{
    /*
        load a new rule to apuc in Plat @index.
    */
	int status = 0;

	uint8_t rule_bar[16];  // 32 bytes for 16 key parts to be organised as (qi, qi#)
	int j ;
	for(j = 0; j <= priority; j ++){ 				// 1-16 bytes of each rule are used as keys (depending on 'priority')
		rule_bar[j] = ~rule[j];
	}
	for(; j < 16 ;j++){ 							//fill the rest of the queries with "don't care"s
		rule[j] = 0xFF;
		rule_bar[j] = 0xFF;
	}
	column_to_wr_rd(index); // write inhibit all coulmns except @index
	{
		for(j=0; j < 8 ;j++){ // load bytes as half-words to VRs
			gvml_cpy_imm_16(DATA_VR+ 2*j, rule[2*j] | (rule[2*j+1] << 8) );
			gvml_cpy_imm_16(DATA_BAR_VR+ 2*j, rule_bar[2*j] | (rule_bar[2*j+1] << 8) );
		}
		uint16_t priority_unar = 0xFFFF >> (15 - priority); 	// convert priority to unary encoding
		gvml_cpy_imm_16(UNARY_PRIORITY, priority_unar); // Load priority to VR
		gvml_set_m(VALID_MRK);                          // Set Valid bit of Plat
	}
	_gvml_write_inhibit_reset();

	return status;
}

// assisting functions:
void bellex_reset_bank(){
	for(int i= 0; i < 24; i++){
		if(i != 15) 			// don't reset index bit!
			gvml_reset_16(i);
	}
	gvml_cpy_16(M4, 15);					// set vr M4 = VR15 (ID )

}

void column_to_wr_rd(int index){
	gvml_reset_16(TEMP);
	gvml_cpy_imm_16(TEMP, index);
	gvml_xor_16(TEMP, M4, TEMP); 			// T1 = T1 ^ ID
	gvml_not_16(TEMP, TEMP); 				// T1 = ~T1. Now T1 = 0xFFFFFF only at the column i
	apl_set_rn_reg(RN_REG_0, TEMP);
	RUN_IMM_FRAG_ASYNC(apl_write_inhbit(RN_REG valid_RN)
	{
		SM_0XFFFF: RL = SB[valid_RN];
		SM_0XFFFF: GL = RL;

		SM_0XFFFF: SB[valid_RN] = GL;
		{
			SM_0XFFFF: RL = SB[valid_RN];
			SM_0XFFFF: RWINH_SET; // enable read inhibit(rwinhset)
		}
	});
}

void _gvml_write_inhibit_reset(){
	RUN_IMM_FRAG_ASYNC(apl_write_inhbit_rst ( RN_REG vr = RN_REG_7){
		SM_0XFFFF: RWINH_RST; // enable read inhibit(rwinhset)
	});
}
