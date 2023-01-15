from torch import nn

from nemo.core.classes import Serialization, Typing, typecheck

class Seq2SeqLoss(Serialization, Typing):

    def __init__(self, blank_index=0, ctc_weight = 0.1, zero_infinity=False, reduction='mean_batch'):

        if reduction not in ['none', 'mean', 'sum', 'mean_batch', 'mean_volume']:
            raise ValueError('`reduction` must be one of [mean, sum, mean_batch, mean_volume]')

        self.config_reduction = reduction
        if reduction == 'mean_batch' or reduction == 'mean_volume':
            ctc_reduction = 'none'
            self._apply_reduction = True
        elif reduction in ['sum', 'mean', 'none']:
            ctc_reduction = reduction
            self._apply_reduction = False
        
        self.ctc_loss = nn.CTCLoss(blank=blank_index, reduction=ctc_reduction, zero_infinity=zero_infinity)
        self.seq_loss = nn.CrossEntropyLoss()
        self.ctc_weight = ctc_weight


    def reduce(self, losses, target_lengths):
        if self.config_reduction == 'mean_batch':
            losses = losses.mean()  # global batch size average
        elif self.config_reduction == 'mean_volume':
            losses = losses.sum() / target_lengths.sum()  # same as above but longer samples weigh more

        return losses

    @typecheck()
    def forward(self, log_probs, logits, encoder_targets, decoder_targets, 
                input_lengths, encoder_target_lengths, decoder_target_lengths):
        """Linear combiniation of two losses, CTCLoss and Autoregressive
           B: Batch size
           T: Time step real output
           T': Time step of the crossponding CTC output
           T": Time step of the input signal
           C: number of Class
           Note that T' > T and T" > T'

        Args:
            log_probs (torch.Tensor): logSoftmax output of the ctc portion of the model with shape [B, T', C]
            logits (torch.Tensor): Softmax output of the autoregressive portion of the model with shape [B, T, C]
            encoder_targets (torch.LongTensor): The target tokens [B, T]
            decoder_targets (torch.LongTensor): The target tokens [B, T]
            input_lengths (torch.LongTensor): The length of input signal [B, T"]
            encoder_target_lengths (torch.LongTensor): The length of target signal [B, T]
            decoder_target_lengths (torch.LongTensor): The length of target signal [B, T]

        Returns:
            torch.Tensor: The value of loss function
        """
        # override forward implementation
        # custom logic, if necessary
        input_lengths = input_lengths.long()
        encoder_target_lengths = encoder_target_lengths.long()
        decoder_target_lengths = decoder_target_lengths.long()

        encoder_targets = encoder_targets.long()
        decoder_targets = decoder_targets.long()
        # here we transpose because we expect [B, T, D] while PyTorch assumes [T, B, D]
        # Pytroch assumption is better :)
        log_probs = log_probs.transpose(1, 0)
        seq_loss = self.seq_loss(input=logits.transpose(1,2), target=decoder_targets)
        ctc_loss = self.ctc_loss(log_probs=log_probs, 
                             targets=encoder_targets,
                             input_lengths=input_lengths, 
                             target_lengths=encoder_target_lengths)

        ctc_loss = self.reduce(ctc_loss, encoder_target_lengths)

        loss = seq_loss + self.ctc_weight * ctc_loss

        return loss, seq_loss, ctc_loss